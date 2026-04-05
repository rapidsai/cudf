# Resume Prompt — Paste this to restart the experiment

```
Continue optimizing the CSV parser for maximum throughput. Resume from the current branch (autoresearch/apr05-csv).

IMPORTANT: The configuration files have been significantly updated since your last run. Re-read ALL of these before doing anything:
- program.md
- CLAUDE.md  
- .claude/rules/discipline.md
- .claude/rules/mcp-plugins.md (new)

Key changes: new primary benchmark target, new logging requirements (AGENT_LOG.md, numbered experiments), research head strategy, updated optimization priority tiers, and a warning about micro-benchmark tunnel vision.

During your research phase, look into these papers and projects for ideas:
- ParPaRaw (arxiv 1905.13415) — GPU CSV parser, compares against cuDF
- RFC 4180 compliant GPU parsing approaches
- GPU text/JSON parsing papers — adjacent techniques that may transfer

The previous 17 experiments reached +75.2% but may have hit a local optimum. The updated config files explain why and what to do about it. Read them carefully and adjust your strategy accordingly.
```
