//! Hash-pinned snapshot of the generated WGSL (meshless/moving-mesh roadmap,
//! governing principle 1: "static path untouched").
//!
//! WHY THIS EXISTS (review-validation #3): the `check-generated-wgsl.yml` CI
//! (`scripts/check_generated_wgsl.sh`) asserts **freshness** — committed files
//! match what the current code regenerates. It does NOT assert **invariance**:
//! if a codegen change perturbs a static model's WGSL, the developer commits
//! the regenerated files and that CI stays green. This test pins the actual
//! CONTENT of every generated WGSL file (FNV-1a 64 hash), so any change to a
//! committed shader — intended or not — fails loudly until it is deliberately
//! blessed.
//!
//! UPDATE RITUAL (deliberate changes only):
//!   1. Make your codegen/model change, run `cargo build` (regenerates WGSL).
//!   2. Inspect `git diff src/solver/gpu/shaders/generated/` — every changed
//!      file must be explained by your change (new `*_<model>` files for a new
//!      model; content diffs ONLY for models whose numerics you intended to
//!      change — static models must stay byte-identical unless the changeset
//!      explicitly declares otherwise, per AGENTS.md).
//!   3. Re-bless: `CFD2_BLESS_WGSL_SNAPSHOT=1 cargo test --test
//!      static_wgsl_snapshot_test` rewrites `tests/static_wgsl_snapshot.txt`.
//!   4. Commit the snapshot file TOGETHER with the shader diffs, and call out
//!      any static-model hash change in the changeset report.
//!
//! No feature gate: this reads committed files only, so it runs under plain
//! `cargo test` and any feature combination.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SNAPSHOT_PATH: &str = "tests/static_wgsl_snapshot.txt";
const GENERATED_DIR: &str = "src/solver/gpu/shaders/generated";

/// FNV-1a 64-bit — dependency-free, stable, plenty for change detection (this
/// is a tamper-evident pin against accidental drift, not a security boundary).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// All committed generated WGSL files, relative path -> content hash.
fn current_hashes() -> BTreeMap<String, u64> {
    let dir = repo_root().join(GENERATED_DIR);
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("read generated WGSL dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
                let rel = path
                    .strip_prefix(&dir)
                    .expect("strip prefix")
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes = std::fs::read(&path).expect("read wgsl file");
                out.insert(rel, fnv1a64(&bytes));
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no generated WGSL files found under {GENERATED_DIR} — run `cargo build` first"
    );
    out
}

fn snapshot_file() -> PathBuf {
    repo_root().join(SNAPSHOT_PATH)
}

fn write_snapshot(hashes: &BTreeMap<String, u64>) {
    let mut body = String::from(
        "# Generated-WGSL content snapshot (FNV-1a 64 per file).\n\
         # Asserts INVARIANCE of committed shaders (the check-generated-wgsl CI only\n\
         # asserts freshness). Update ONLY deliberately:\n\
         #   CFD2_BLESS_WGSL_SNAPSHOT=1 cargo test --test static_wgsl_snapshot_test\n\
         # and see the update ritual in tests/static_wgsl_snapshot_test.rs.\n",
    );
    for (name, hash) in hashes {
        body.push_str(&format!("{hash:016x}  {name}\n"));
    }
    std::fs::write(snapshot_file(), body).expect("write snapshot file");
}

fn read_snapshot(path: &Path) -> BTreeMap<String, u64> {
    let body = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing WGSL snapshot file {SNAPSHOT_PATH} ({e}); bless it once with \
             CFD2_BLESS_WGSL_SNAPSHOT=1 cargo test --test static_wgsl_snapshot_test"
        )
    });
    let mut out = BTreeMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (hash, name) = line
            .split_once("  ")
            .unwrap_or_else(|| panic!("malformed snapshot line: {line:?}"));
        let hash = u64::from_str_radix(hash, 16)
            .unwrap_or_else(|e| panic!("malformed snapshot hash in {line:?}: {e}"));
        out.insert(name.to_string(), hash);
    }
    out
}

#[test]
fn generated_wgsl_matches_snapshot() {
    let current = current_hashes();

    if std::env::var("CFD2_BLESS_WGSL_SNAPSHOT").as_deref() == Ok("1") {
        write_snapshot(&current);
        println!(
            "[wgsl-snapshot] BLESSED {} files into {SNAPSHOT_PATH}",
            current.len()
        );
        return;
    }

    let pinned = read_snapshot(&snapshot_file());

    let mut changed: Vec<&String> = Vec::new();
    let mut missing: Vec<&String> = Vec::new();
    for (name, hash) in &pinned {
        match current.get(name) {
            Some(cur) if cur == hash => {}
            Some(_) => changed.push(name),
            None => missing.push(name),
        }
    }
    let added: Vec<&String> = current
        .keys()
        .filter(|name| !pinned.contains_key(*name))
        .collect();

    assert!(
        changed.is_empty() && missing.is_empty() && added.is_empty(),
        "generated WGSL drifted from the pinned snapshot.\n\
         changed ({} files): {changed:?}\n\
         removed ({} files): {missing:?}\n\
         added   ({} files): {added:?}\n\
         If (and only if) this is deliberate, follow the update ritual in \
         tests/static_wgsl_snapshot_test.rs (inspect git diff, then \
         CFD2_BLESS_WGSL_SNAPSHOT=1).",
        changed.len(),
        missing.len(),
        added.len(),
    );
    println!(
        "[wgsl-snapshot] {} generated WGSL files match the pinned snapshot",
        current.len()
    );
}
