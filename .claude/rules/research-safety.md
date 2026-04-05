---
paths:
  - "**"
---

# Research Safety: Read-Only Web Access

You have **unlimited web search budget**. Use it aggressively to find papers, techniques, and optimization ideas.

## Allowed

- Search for academic papers, blog posts, NVIDIA guides, CUDA documentation
- Read code examples online to understand concepts and algorithms
- Read API docs for CUB, Thrust, CUDA runtime
- Read how other GPU databases solve similar problems

## Forbidden

- `curl | bash`, `wget`, `git clone`, `pip install`, `npm install` — or any command that downloads and runs external code
- Copy-pasting code snippets from the web and executing them as scripts
- Adding external dependencies not already in the project

## Why

External code may have incompatible licenses, vulnerabilities, or dependencies that break the build. Understanding techniques and reimplementing them from scratch ensures the code fits cuDF's architecture, style, and performance constraints. Your implementations will be better because you understand the full context.
