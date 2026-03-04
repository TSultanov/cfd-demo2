/**
 * Codebase Review Extension
 *
 * Bundles the entire project source into a Gemini context cache, then lets you
 * ask review questions cheaply against that cached context.
 *
 * Commands:
 *   /review-init   – collect files, count tokens, create cache
 *   /review <q>    – ask a question against the cached codebase
 *   /review-status – show cache details
 *   /review-clear  – delete the cache
 *
 * API key resolution (in order):
 *   1. Pi model registry (Google provider — /login google, models.json, GOOGLE_API_KEY)
 *   2. GEMINI_API_KEY environment variable
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { getMarkdownTheme } from "@mariozechner/pi-coding-agent";
import { Text, Container, Spacer, Markdown } from "@mariozechner/pi-tui";

// ─── Configuration ─────────────────────────────────────────────────────────────

const DEFAULT_MODEL = "gemini-3.1-pro-preview";
const API_BASE = "https://generativelanguage.googleapis.com/v1beta";

/** Files/dirs to always exclude (relative to project root). */
const DEFAULT_EXCLUDES = [
  "target/",
  ".git/",
  ".idea/",
  ".code/",
  ".kimi/",
  ".pi/",
  "node_modules/",
  "bindings.rs",    // auto-generated wgsl_bindgen output
];

/** File extensions to include. */
const SOURCE_EXTENSIONS = new Set([".rs", ".wgsl", ".toml", ".md"]);

/** Only include .md files at project root (Cargo.toml, README, etc.) */
const MD_ROOT_ONLY = true;

/** How aggressively to strip content.
 *  - "aggressive": blanks, comments (keep doc comments), #[cfg(test)] blocks
 *  - "moderate":   blanks, #[cfg(test)] blocks
 *  - "none":       raw files
 */
type StripLevel = "aggressive" | "moderate" | "none";

// ─── Gemini API helpers ────────────────────────────────────────────────────────

async function geminiPost(apiKey: string, endpoint: string, body: unknown): Promise<any> {
  const url = `${API_BASE}${endpoint}${endpoint.includes("?") ? "&" : "?"}key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Gemini API ${res.status}: ${text}`);
  }
  return res.json();
}

async function geminiDelete(apiKey: string, endpoint: string): Promise<void> {
  const url = `${API_BASE}${endpoint}?key=${apiKey}`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Gemini API ${res.status}: ${text}`);
  }
}

async function countTokens(apiKey: string, model: string, text: string): Promise<number> {
  const data = await geminiPost(apiKey, `/models/${model}:countTokens`, {
    contents: [{ parts: [{ text }] }],
  });
  return data.totalTokens ?? 0;
}

async function createCache(
  apiKey: string,
  model: string,
  systemInstruction: string,
  codebaseText: string,
  ttlSeconds: number,
): Promise<{ name: string; usageMetadata: any; expireTime: string }> {
  const body: any = {
    model: `models/${model}`,
    displayName: "cfd2-codebase-review",
    systemInstruction: { parts: [{ text: systemInstruction }] },
    contents: [
      {
        role: "user",
        parts: [{ text: codebaseText }],
      },
    ],
    ttl: `${ttlSeconds}s`,
  };
  return geminiPost(apiKey, "/cachedContents", body);
}

async function generateWithCache(
  apiKey: string,
  model: string,
  cacheName: string,
  question: string,
  maxOutputTokens: number,
  signal?: AbortSignal,
): Promise<AsyncGenerator<string>> {
  const url = `${API_BASE}/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cachedContent: cacheName,
      contents: [{ role: "user", parts: [{ text: question }] }],
      generationConfig: { maxOutputTokens },
    }),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Gemini API ${res.status}: ${text}`);
  }

  async function* streamChunks(): AsyncGenerator<string> {
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // Parse SSE lines
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") return;
          try {
            const parsed = JSON.parse(payload);
            const parts = parsed?.candidates?.[0]?.content?.parts;
            if (parts) {
              for (const part of parts) {
                if (part.text) yield part.text;
              }
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  return streamChunks();
}

// ─── File collection & stripping ───────────────────────────────────────────────

function shouldExclude(relPath: string, excludes: string[]): boolean {
  for (const ex of excludes) {
    if (ex.endsWith("/")) {
      // directory prefix
      if (relPath.startsWith(ex) || relPath.includes(`/${ex}`)) return true;
    } else if (ex.includes("*") || ex.includes("?")) {
      // glob pattern
      if ((path as any).matchesGlob(relPath, ex)) return true;
    } else {
      // filename match
      if (relPath.endsWith(`/${ex}`) || relPath === ex || path.basename(relPath) === ex) return true;
    }
  }
  return false;
}

function collectSourceFiles(
  root: string,
  excludes: string[],
): { relPath: string; absPath: string }[] {
  const results: { relPath: string; absPath: string }[] = [];

  function walk(dir: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of entries) {
      const abs = path.join(dir, ent.name);
      const rel = path.relative(root, abs);
      if (shouldExclude(rel, excludes)) continue;

      if (ent.isDirectory()) {
        walk(abs);
      } else if (ent.isFile()) {
        const ext = path.extname(ent.name);
        if (!SOURCE_EXTENSIONS.has(ext)) continue;
        // Only include .md at root level
        if (ext === ".md" && MD_ROOT_ONLY && path.dirname(abs) !== root) continue;
        // Only include .toml files named Cargo.toml
        if (ext === ".toml" && ent.name !== "Cargo.toml") continue;
        results.push({ relPath: rel, absPath: abs });
      }
    }
  }

  walk(root);
  results.sort((a, b) => a.relPath.localeCompare(b.relPath));
  return results;
}

function stripContent(content: string, ext: string, level: StripLevel): string {
  if (level === "none") return content;

  const lines = content.split("\n");
  const stripped: string[] = [];
  let inTestBlock = false;
  let testBraceDepth = 0;
  let inBlockComment = false;

  for (const line of lines) {
    const trimmed = line.trimStart();

    // Track block comments for .rs files
    if ((ext === ".rs" || ext === ".wgsl") && !inTestBlock) {
      if (inBlockComment) {
        if (trimmed.includes("*/")) {
          inBlockComment = false;
        }
        if (level === "aggressive") continue;
      }
      if (trimmed.startsWith("/*") && !trimmed.includes("*/")) {
        inBlockComment = true;
        if (level === "aggressive") continue;
      }
      if (trimmed.startsWith("/*") && trimmed.includes("*/")) {
        if (level === "aggressive") continue;
      }
    }

    // Strip #[cfg(test)] blocks for .rs files
    if (ext === ".rs" && trimmed.startsWith("#[cfg(test)]")) {
      inTestBlock = true;
      testBraceDepth = 0;
      continue;
    }
    if (inTestBlock) {
      testBraceDepth += (line.match(/{/g) ?? []).length;
      testBraceDepth -= (line.match(/}/g) ?? []).length;
      if (testBraceDepth <= 0 && line.includes("}")) {
        inTestBlock = false;
      }
      continue;
    }

    // Skip blank lines
    if (!trimmed) continue;

    // Aggressive: skip standalone comments (keep doc comments ///, //!)
    if (level === "aggressive" && (ext === ".rs" || ext === ".wgsl")) {
      if (
        trimmed.startsWith("//") &&
        !trimmed.startsWith("///") &&
        !trimmed.startsWith("//!")
      ) {
        continue;
      }
    }

    stripped.push(line);
  }

  return stripped.join("\n");
}

function packCodebase(
  root: string,
  files: { relPath: string; absPath: string }[],
  level: StripLevel,
): { packed: string; fileCount: number; originalBytes: number; packedBytes: number } {
  const parts: string[] = [];
  let originalBytes = 0;
  let fileCount = 0;

  for (const f of files) {
    let content: string;
    try {
      content = fs.readFileSync(f.absPath, "utf-8");
    } catch {
      continue;
    }
    originalBytes += Buffer.byteLength(content, "utf-8");
    const ext = path.extname(f.relPath);
    const stripped = stripContent(content, ext, level);
    parts.push(`=== FILE: ${f.relPath} ===\n${stripped}`);
    fileCount++;
  }

  const packed = parts.join("\n\n");
  return {
    packed,
    fileCount,
    originalBytes,
    packedBytes: Buffer.byteLength(packed, "utf-8"),
  };
}

// ─── System prompt for the review context ──────────────────────────────────────

function buildSystemPrompt(): string {
  return `You are an expert code reviewer and software architect analyzing a Rust + WGSL codebase.

This is "cfd2" — a 2D Computational Fluid Dynamics solver for incompressible and compressible laminar flow, implemented in Rust with GPU compute via wgpu/WebGPU.

Architecture highlights:
- Rust host code manages mesh generation, solver orchestration, and a UI (egui)
- WGSL compute shaders run the actual numerical kernels on GPU
- A custom IR (cfd2_ir) and codegen pipeline (cfd2_codegen) generates optimized WGSL kernels
- Kernel fusion system merges multiple operations into fewer GPU dispatches
- Linear solvers (FGMRES, AMG, CG) run entirely on GPU
- Port-based field access system with typed dimensions
- OpenFOAM reference test suite for validation

The entire codebase is provided below. When answering questions:
- Reference specific files and line ranges
- Cite function/struct/module names precisely
- Identify patterns, anti-patterns, and potential issues
- Be thorough but structured in your analysis
- Use markdown formatting for readability`;
}

// ─── Extension ─────────────────────────────────────────────────────────────────

interface CacheState {
  name: string;
  model: string;
  tokenCount: number;
  fileCount: number;
  expireTime: string;
  createdAt: number;
  ttlSeconds: number;
}

export default function (pi: ExtensionAPI) {
  let cache: CacheState | null = null;

  /** Notify user — works in all modes (falls back to console.log in print/json) */
  function notify(ctx: any, msg: string, level: "info" | "warning" | "error" = "info") {
    if (ctx.hasUI) {
      ctx.ui.notify(msg, level);
    } else {
      const prefix = level === "error" ? "ERROR: " : level === "warning" ? "WARN: " : "";
      console.log(prefix + msg);
    }
  }

  /**
   * Resolve the Gemini API key. Checks (in order):
   * 1. Pi model registry (Google provider — covers /login OAuth, models.json, GOOGLE_API_KEY)
   * 2. GEMINI_API_KEY env var (explicit override)
   */
  async function resolveApiKey(ctx: any): Promise<string> {
    // Try Pi's model registry first — handles OAuth, models.json apiKey, and
    // the built-in GOOGLE_API_KEY env-var resolution for the "google" provider.
    const piKey = await ctx.modelRegistry.getApiKeyForProvider("google");
    if (piKey) return piKey;

    // Fallback: explicit GEMINI_API_KEY env var
    const envKey = process.env.GEMINI_API_KEY;
    if (envKey) return envKey;

    throw new Error(
      "No Gemini API key found. Configure one via:\n" +
      "  • Pi: /login google, or set GOOGLE_API_KEY\n" +
      "  • Env: export GEMINI_API_KEY=...",
    );
  }

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)}MB`;
  }

  function formatTokens(n: number): string {
    if (n < 1000) return `${n}`;
    if (n < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
    return `${(n / 1_000_000).toFixed(2)}M`;
  }

  function isCacheValid(): boolean {
    if (!cache) return false;
    return new Date(cache.expireTime).getTime() > Date.now();
  }

  // ── /review-init ──────────────────────────────────────────────────────────

  pi.registerCommand("review-init", {
    description:
      "Build a Gemini context cache of the codebase. Args: [--model MODEL] [--strip aggressive|moderate|none] [--ttl SECONDS] [--exclude PATTERN,...]",
    handler: async (args, ctx) => {
      // Parse arguments
      const tokens = (args ?? "").split(/\s+/).filter(Boolean);
      let model = DEFAULT_MODEL;
      let stripLevel: StripLevel = "aggressive";
      let ttlSeconds = 3600;
      let extraExcludes: string[] = [];

      for (let i = 0; i < tokens.length; i++) {
        switch (tokens[i]) {
          case "--model":
            model = tokens[++i] ?? model;
            break;
          case "--strip":
            stripLevel = (tokens[++i] as StripLevel) ?? stripLevel;
            break;
          case "--ttl":
            ttlSeconds = parseInt(tokens[++i] ?? "3600", 10);
            break;
          case "--exclude":
            extraExcludes = (tokens[++i] ?? "").split(",").filter(Boolean);
            break;
        }
      }

      const excludes = [...DEFAULT_EXCLUDES, ...extraExcludes];

      ctx.ui.setStatus("review", "Resolving API key...");

      try {
        const apiKey = await resolveApiKey(ctx);

        // 1. Collect files
        ctx.ui.setStatus("review", "Collecting source files...");
        const files = collectSourceFiles(ctx.cwd, excludes);
        ctx.ui.setStatus("review", `Packing ${files.length} files (strip: ${stripLevel})...`);

        // 2. Pack
        const { packed, fileCount, originalBytes, packedBytes } = packCodebase(
          ctx.cwd,
          files,
          stripLevel,
        );

        notify(ctx, 
          `Packed ${fileCount} files: ${formatSize(originalBytes)} → ${formatSize(packedBytes)}`,
          "info",
        );

        // 3. Count tokens
        ctx.ui.setStatus("review", "Counting tokens via Gemini API...");
        const systemPrompt = buildSystemPrompt();
        const combinedText = systemPrompt + "\n\n" + packed;
        const tokenCount = await countTokens(apiKey, model, combinedText);

        notify(ctx, `Token count: ${formatTokens(tokenCount)} (model: ${model})`, "info");

        // Check if it fits (leave room for output)
        if (tokenCount > 950_000) {
          ctx.ui.setStatus("review", undefined);
          notify(ctx, 
            `⚠️ ${formatTokens(tokenCount)} tokens may exceed the context window. ` +
              `Try --strip aggressive, --exclude to remove more files, or a model with a larger window.`,
            "warning",
          );

          // Show biggest files to help user decide what to exclude
          const fileSizes = files.map((f) => {
            try {
              const stat = fs.statSync(f.absPath);
              return { rel: f.relPath, size: stat.size };
            } catch {
              return { rel: f.relPath, size: 0 };
            }
          });
          fileSizes.sort((a, b) => b.size - a.size);
          const top = fileSizes.slice(0, 15);
          const hint = top
            .map((f) => `  ${formatSize(f.size).padStart(8)} ${f.rel}`)
            .join("\n");
          notify(ctx, `Largest files:\n${hint}`, "info");

          const ok = ctx.hasUI
            ? await ctx.ui.confirm(
                "Token count is high",
                `${formatTokens(tokenCount)} tokens. Continue creating cache anyway?`,
              )
            : false;
          if (!ok) return;
        }

        // 4. Create cache
        ctx.ui.setStatus("review", "Creating Gemini context cache...");
        const result = await createCache(apiKey, model, systemPrompt, packed, ttlSeconds);

        cache = {
          name: result.name,
          model,
          tokenCount,
          fileCount,
          expireTime: result.expireTime,
          createdAt: Date.now(),
          ttlSeconds,
        };

        const expiry = new Date(result.expireTime);
        const minutesLeft = Math.round((expiry.getTime() - Date.now()) / 60000);

        ctx.ui.setStatus("review", `Cache ready (${formatTokens(tokenCount)} tokens)`);
        notify(ctx, 
          `✅ Cache created!\n` +
            `  Name: ${result.name}\n` +
            `  Model: ${model}\n` +
            `  Tokens: ${formatTokens(tokenCount)}\n` +
            `  Files: ${fileCount}\n` +
            `  Expires in: ${minutesLeft} min\n` +
            `  Use /review <question> to query`,
          "info",
        );
      } catch (err: any) {
        ctx.ui.setStatus("review", undefined);
        notify(ctx, `❌ review-init failed: ${err.message}`, "error");
      }
    },
  });

  // ── /review ───────────────────────────────────────────────────────────────

  pi.registerCommand("review", {
    description: "Ask a review question against the cached codebase. Usage: /review <question>",
    handler: async (args, ctx) => {
      if (!args?.trim()) {
        notify(ctx, "Usage: /review <your question about the codebase>", "warning");
        return;
      }

      if (!cache || !isCacheValid()) {
        notify(ctx, 
          "No active cache. Run /review-init first to build the codebase cache.",
          "warning",
        );
        return;
      }

      const question = args.trim();

      try {
        const apiKey = await resolveApiKey(ctx);
        ctx.ui.setStatus("review", "Querying Gemini...");

        const stream = await generateWithCache(
          apiKey,
          cache.model,
          cache.name,
          question,
          65536, // max output tokens
        );

        let fullResponse = "";
        let chunks = 0;
        for await (const chunk of stream) {
          fullResponse += chunk;
          chunks++;
          if (chunks % 10 === 0) {
            ctx.ui.setStatus("review", `Receiving... ${(fullResponse.length / 1024).toFixed(0)}KB`);
          }
        }

        ctx.ui.setStatus(
          "review",
          `Cache ready (${formatTokens(cache.tokenCount)} tokens)`,
        );

        if (!fullResponse.trim()) {
          notify(ctx, "⚠️ Gemini returned an empty response.", "warning");
          return;
        }

        // Save response to file
        const outputDir = path.resolve(ctx.cwd, "target");
        if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
        const outputPath = path.resolve(outputDir, "review-response.md");
        const header =
          `# Review: ${question}\n\n` +
          `Model: ${cache.model} | Cached tokens: ${formatTokens(cache.tokenCount)} | ${new Date().toISOString()}\n\n---\n\n`;
        fs.writeFileSync(outputPath, header + fullResponse, "utf-8");

        // Show result inline and persist to session
        notify(ctx, `✅ Review complete (${(fullResponse.length / 1024).toFixed(1)}KB). Saved to ${outputPath}`, "info");

        pi.sendMessage(
          {
            customType: "codebase-review",
            content: fullResponse,
            display: true,
            details: {
              question,
              model: cache.model,
              cachedTokens: cache.tokenCount,
              outputPath,
            },
          },
          { triggerTurn: false },
        );
      } catch (err: any) {
        ctx.ui.setStatus(
          "review",
          cache ? `Cache ready (${formatTokens(cache.tokenCount)} tokens)` : undefined,
        );
        notify(ctx, `❌ Review query failed: ${err.message}`, "error");
      }
    },
  });

  // ── /review-status ────────────────────────────────────────────────────────

  pi.registerCommand("review-status", {
    description: "Show current codebase review cache status",
    handler: async (_args, ctx) => {
      if (!cache) {
        notify(ctx, "No cache. Run /review-init to create one.", "info");
        return;
      }

      const expiry = new Date(cache.expireTime);
      const valid = expiry.getTime() > Date.now();
      const minutesLeft = Math.max(0, Math.round((expiry.getTime() - Date.now()) / 60000));

      notify(ctx, 
        `Codebase Review Cache\n` +
          `  Status: ${valid ? "✅ Active" : "❌ Expired"}\n` +
          `  Cache ID: ${cache.name}\n` +
          `  Model: ${cache.model}\n` +
          `  Tokens: ${formatTokens(cache.tokenCount)}\n` +
          `  Files: ${cache.fileCount}\n` +
          `  Expires: ${valid ? `in ${minutesLeft} min` : "expired"}\n` +
          `  TTL: ${cache.ttlSeconds}s`,
        "info",
      );
    },
  });

  // ── /review-clear ─────────────────────────────────────────────────────────

  pi.registerCommand("review-clear", {
    description: "Delete the codebase review cache",
    handler: async (_args, ctx) => {
      if (!cache) {
        notify(ctx, "No cache to clear.", "info");
        return;
      }

      try {
        const apiKey = await resolveApiKey(ctx);
        await geminiDelete(apiKey, `/${cache.name}`);
        notify(ctx, `✅ Cache ${cache.name} deleted.`, "info");
      } catch (err: any) {
        notify(ctx, `⚠️ Cache deletion failed (may have already expired): ${err.message}`, "warning");
      }

      cache = null;
      ctx.ui.setStatus("review", undefined);
    },
  });

  // ── /review-files ─────────────────────────────────────────────────────────

  pi.registerCommand("review-files", {
    description:
      "List files that would be included in the cache. Args: [--exclude PATTERN,...]",
    handler: async (args, ctx) => {
      const tokens = (args ?? "").split(/\s+/).filter(Boolean);
      let extraExcludes: string[] = [];
      for (let i = 0; i < tokens.length; i++) {
        if (tokens[i] === "--exclude") {
          extraExcludes = (tokens[++i] ?? "").split(",").filter(Boolean);
        }
      }

      const excludes = [...DEFAULT_EXCLUDES, ...extraExcludes];
      const files = collectSourceFiles(ctx.cwd, excludes);

      const byExt = new Map<string, { count: number; bytes: number }>();
      let totalBytes = 0;
      for (const f of files) {
        const ext = path.extname(f.relPath);
        try {
          const stat = fs.statSync(f.absPath);
          const entry = byExt.get(ext) ?? { count: 0, bytes: 0 };
          entry.count++;
          entry.bytes += stat.size;
          byExt.set(ext, entry);
          totalBytes += stat.size;
        } catch {
          /* skip */
        }
      }

      let summary = `Files to cache: ${files.length} (${formatSize(totalBytes)})\n\nBy extension:\n`;
      for (const [ext, info] of [...byExt.entries()].sort((a, b) => b[1].bytes - a[1].bytes)) {
        summary += `  ${ext.padEnd(8)} ${String(info.count).padStart(4)} files  ${formatSize(info.bytes).padStart(8)}\n`;
      }

      // Show top 20 largest files
      const fileSizes = files.map((f) => {
        try {
          return { rel: f.relPath, size: fs.statSync(f.absPath).size };
        } catch {
          return { rel: f.relPath, size: 0 };
        }
      });
      fileSizes.sort((a, b) => b.size - a.size);
      summary += `\nTop 20 largest:\n`;
      for (const f of fileSizes.slice(0, 20)) {
        summary += `  ${formatSize(f.size).padStart(8)} ${f.rel}\n`;
      }

      notify(ctx, summary, "info");
    },
  });

  // ── /review-ask (multi-question batch) ────────────────────────────────────

  pi.registerCommand("review-ask", {
    description:
      "Ask multiple review questions from a file. Usage: /review-ask <questions-file.md>",
    handler: async (args, ctx) => {
      if (!args?.trim()) {
        notify(ctx, "Usage: /review-ask <path-to-questions-file>", "warning");
        return;
      }

      if (!cache || !isCacheValid()) {
        notify(ctx, 
          "No active cache. Run /review-init first.",
          "warning",
        );
        return;
      }

      const questionsPath = path.resolve(ctx.cwd, args.trim());
      let questionsContent: string;
      try {
        questionsContent = fs.readFileSync(questionsPath, "utf-8");
      } catch (err: any) {
        notify(ctx, `Cannot read ${questionsPath}: ${err.message}`, "error");
        return;
      }

      // Parse questions: lines starting with "- " or "## " or numbered "1. "
      const questions: string[] = [];
      for (const line of questionsContent.split("\n")) {
        const trimmed = line.trim();
        if (
          trimmed.startsWith("- ") ||
          trimmed.startsWith("## ") ||
          /^\d+\.\s/.test(trimmed)
        ) {
          const q = trimmed.replace(/^[-#\d.]+\s*/, "").trim();
          if (q.length > 5) questions.push(q);
        }
      }

      if (questions.length === 0) {
        notify(ctx, 
          "No questions found. Use lines starting with '- ', '## ', or '1. '",
          "warning",
        );
        return;
      }

      notify(ctx, `Found ${questions.length} questions. Starting batch review...`, "info");

      let apiKey: string;
      try {
        apiKey = await resolveApiKey(ctx);
      } catch (err: any) {
        notify(ctx, `❌ ${err.message}`, "error");
        return;
      }

      const outputPath = path.resolve(ctx.cwd, "target/review-output.md");
      const outputDir = path.dirname(outputPath);
      if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

      let output = `# Codebase Review\n\nModel: ${cache.model}\nCached tokens: ${formatTokens(cache.tokenCount)}\nDate: ${new Date().toISOString()}\n\n---\n\n`;

      for (let i = 0; i < questions.length; i++) {
        const q = questions[i];
        ctx.ui.setStatus("review", `Question ${i + 1}/${questions.length}: ${q.slice(0, 40)}...`);

        try {
          const stream = await generateWithCache(
            apiKey,
            cache.model,
            cache.name,
            q,
            65536,
          );

          let answer = "";
          for await (const chunk of stream) {
            answer += chunk;
          }

          output += `## Q${i + 1}: ${q}\n\n${answer}\n\n---\n\n`;
        } catch (err: any) {
          output += `## Q${i + 1}: ${q}\n\n**ERROR:** ${err.message}\n\n---\n\n`;
        }
      }

      fs.writeFileSync(outputPath, output, "utf-8");

      ctx.ui.setStatus(
        "review",
        `Cache ready (${formatTokens(cache.tokenCount)} tokens)`,
      );
      notify(ctx, 
        `✅ Batch review complete!\n  ${questions.length} questions answered\n  Output: ${outputPath}`,
        "info",
      );

      // Also inject summary into session
      pi.sendMessage(
        {
          customType: "codebase-review",
          content: `Batch review complete. ${questions.length} questions answered.\nFull output saved to: ${outputPath}`,
          display: true,
          details: { batchFile: questionsPath, outputFile: outputPath, questionCount: questions.length },
        },
        { triggerTurn: false },
      );
    },
  });

  // ── Message renderer ──────────────────────────────────────────────────────

  pi.registerMessageRenderer("codebase-review", (message, { expanded }, theme) => {
    const details = message.details as any;
    const mdTheme = getMarkdownTheme();
    const container = new Container();

    let header = theme.fg("accent", theme.bold("📋 Codebase Review"));
    if (details?.model) header += theme.fg("muted", ` (${details.model})`);
    if (details?.cachedTokens)
      header += theme.fg("dim", ` [${formatTokens(details.cachedTokens)} cached tokens]`);
    container.addChild(new Text(header, 0, 0));

    if (details?.question) {
      container.addChild(new Text(theme.fg("muted", `Q: ${details.question}`), 0, 0));
      container.addChild(new Spacer(1));
    }

    const content = typeof message.content === "string" ? message.content : "";
    if (expanded) {
      // Full markdown rendering
      container.addChild(new Markdown(content, 0, 0, mdTheme));
    } else {
      // Show first 20 lines collapsed
      const lines = content.split("\n");
      const preview = lines.slice(0, 20).join("\n");
      container.addChild(new Markdown(preview, 0, 0, mdTheme));
      if (lines.length > 20) {
        container.addChild(
          new Text(theme.fg("muted", `\n... ${lines.length - 20} more lines (Ctrl+O to expand)`), 0, 0),
        );
      }
    }

    return container;
  });
}
