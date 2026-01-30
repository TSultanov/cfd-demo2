# AGENTS.md instructions for /Volumes/sources/cfd2
## Using `kimi` as a delegated coding agent

`kimi` is installed and can be used as a “sub-agent” to implement scoped code changes. The pattern is an outer loop (you) that plans and validates, and an inner loop (`kimi`) that executes a single iteration of that plan.

### Core loop (plan → run → review → update plan → repeat)

Treat the “overarching plan” file path as a parameter (it is `PORT_REFACTOR_PLAN.md` today, but may change later):

1. **Choose the plan slice**
   - Read the overarching plan file (`$PLAN_FILE`) and pick the next *small, coherent* chunk of work (aim for ≤1–3 related checklist items).
   - Scan the repo for relevant files/symbols so the iteration plan matches reality (use `rg`, `cargo test`, etc.).

2. **Write an iteration plan (to feed to `kimi`)**
   - Create a prompt file (recommended under `.kimi/`) that includes:
     - Context: user prompt + goal.
     - The overarching plan file path (`$PLAN_FILE`) as the source of truth (but **do not** let `kimi` edit it).
     - A step-by-step implementation plan for *this* iteration.
     - Acceptance criteria (what “done” means).
     - Validation commands to run (tests/build checks).
     - Output format request (summary + files changed + commands/tests run + suggested plan updates).

3. **Run `kimi` with the iteration plan**
   - Non-interactive mode (efficient for long prompts): `--print` reads stdin, but note it **implicitly enables auto-approve**.
   - Recommended invocation:
     ```bash
     mkdir -p .kimi/prompts .kimi/logs
     PLAN_FILE=PORT_REFACTOR_PLAN.md   # set to whichever overarching plan is active
     ITER=001

     # Write the prompt in: .kimi/prompts/iter-$ITER.md
     cat .kimi/prompts/iter-$ITER.md | kimi --print --input-format text --work-dir . --continue --max-steps-per-turn 40 --quiet | tee .kimi/logs/iter-$ITER.txt
     ```
   - For more verbose logs (tool calls, intermediate steps), drop `--quiet` (or add `--verbose`).
   - If you need manual approvals instead of auto-approve, run `kimi` without `--print` (interactive mode) and paste the prompt.
   - If `kimi` starts to drift or forget context, drop `--continue` to start a fresh session (or use `--session ...` explicitly).

4. **Review and validate `kimi`’s work (you do this)**
   - Inspect changes:
     - `git status --porcelain`
     - `git diff --stat`
     - `git diff`
   - Run the validation commands from the iteration plan (minimum: `cargo test`).
   - If anything is off-scope or incorrect:
     - Either fix it directly, or
     - Write a follow-up `kimi` prompt scoped only to the corrections.

5. **Update the overarching plan file (you do this)**
   - Edit `$PLAN_FILE` to:
     - Mark completed checklist items.
     - Note what changed (briefly) and any follow-ups discovered during review.
     - Adjust progress/status tables if present.
   - Use `kimi`’s “suggested plan updates” section as input, but keep this edit manual/reviewed.

6. **Repeat**
   - Use the updated `$PLAN_FILE` + repo state to generate the next iteration plan and run `kimi` again.

### Prompt template (copy/paste)

Put this in `.kimi/prompts/iter-XXX.md` and fill in the bracketed parts:

```markdown
You are a coding agent working in this repository.

Overarching plan file: [PATH TO PLAN FILE]
Important: Do NOT edit the plan file. I will update it after review.

# Goal
[User prompt in 1–3 sentences]

# Repo context (brief)
- Current branch/commit: [optional]
- Relevant modules/files: [bullets]

# Iteration scope (do only this)
- [Checklist item 1]
- [Checklist item 2]

# Implementation plan (follow in order; stop and report if blocked)
1) [...]
2) [...]

# Acceptance criteria
- [...]
- [...]

# Validation (run these and report pass/fail)
- cargo test
- [any targeted tests]

# Constraints
- Keep changes minimal and on-scope.
- No unrelated refactors or formatting-only diffs.
- Don’t run destructive commands (no `git reset --hard`, no `rm -rf`, etc.).
- Don’t commit or push.

# Output format
1) Summary (what you changed and why)
2) Files changed
3) Commands/tests run + results
4) Suggested updates to the overarching plan (bullet list; do not edit the file)
```
