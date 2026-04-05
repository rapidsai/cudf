# MCP & Plugin Installation Policy

The agent may install MCP servers or CLI plugins on the fly if they genuinely help with research, profiling, benchmarking, or analysis.

## Allowed

- Install well-known, trusted MCP servers (e.g., documentation lookup, profiling tools, analysis helpers)
- Install CLI tools available via system package managers if they aid benchmarking or profiling
- Configure MCP servers in `.claude/settings.json` or equivalent

## When User Intervention Is Needed

If an MCP server or plugin requires **user authentication or manual setup** (OAuth tokens, API keys, SSH keys, browser-based auth, etc.):

1. Write the setup instructions to `MCP_SETUP_NEEDED.md` in the project root
2. Include: tool name, what it does, why it's useful, and exact setup steps
3. Continue with the experiment loop — don't block on the missing tool
4. The user will configure it before the next session

## Forbidden

- Installing random or unverified MCP servers from unknown sources
- Any MCP/plugin that downloads and executes external optimization code
- Anything that violates the read-only web research rule (research-safety.md)
