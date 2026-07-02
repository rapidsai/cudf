---
paths:
  - "**"
---

# Autoresearch Rules

All experiment rules, safety constraints, discipline guidelines, and revert procedures are consolidated in `program.md`. Read it before making any changes.

Key safety reminders (details in program.md):
- NEVER use `git reset` (any mode) or `git rebase` — use `git revert HEAD --no-edit`
- NEVER modify `cpp/benchmarks/`, `cpp/tests/`, or `eval.sh`
- NEVER edit or delete existing content in AGENT_LOG.md or results.tsv — append only
- NEVER use `git add -A` or `git add .` — stage only code files explicitly
- Always pass tests + `--profile` crash check before benchmarking
