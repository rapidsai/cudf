---
paths:
  - "cpp/src/**"
---

# Experiment Discipline Rules

These prevent drift during long-running autonomous GPU optimization sessions.

## Rule 1 — Stay on objective
Every experiment must directly target the optimization goal for the current module. If you're thinking "this is interesting, let me explore..." — STOP. Log the idea in results.tsv as an `idea` status row and move on.

## Rule 2 — One idea per experiment
Related changes (e.g. a new algorithm + the memory layout it requires) belong together. Unrelated ideas must be separate experiments. Each build-test-benchmark cycle costs 10-30 minutes — if an experiment with mixed ideas fails, you won't know which idea caused it and you've wasted that entire cycle.

## Rule 3 — Establish baseline noise floor
On your first run, run the baseline benchmark **3 times** without code changes. Record the variance in key metrics (Elem/s, time). Any future improvement smaller than this variance is benchmark noise, not a real improvement. Do not keep changes within the noise floor.

## Rule 4 — Verify surprising results
If a result looks too good (>20% gain from a minor change), re-run the benchmark twice more. NVBench measurements have natural variance from GPU thermal throttling, memory allocation timing, and other factors. Only trust reproducible numbers across multiple runs.

## Rule 5 — Clean state between experiments
After every experiment (keep or discard), delete temporary files: `rm -f build.log test.log run.log`. Stale logs from prior experiments accumulate in context and contribute to drift over many iterations.

## Rule 6 — Circuit breaker: deep research mode on 3 consecutive failures
If you hit 3 consecutive `discard` or `crash` results, you are going down the wrong path. STOP coding entirely and enter **deep research mode**:
1. Re-read the source code from disk (not from memory — your cached understanding may be wrong after many edits)
2. Re-read results.tsv AND AGENT_LOG.md end-to-end to see what's been tried and what was learned
3. Spawn **2-3 researcher agents** with the full experiment history (pass them results.tsv and AGENT_LOG.md contents). Each researcher should focus on a different optimization direction
4. Do NOT proceed to the next experiment until you have a genuinely solid idea with **clear theoretical justification** (GPU architecture reasoning, paper backing, or profiling-based evidence)
5. Invest significantly more time in research than you normally would — one well-researched experiment is worth more than five speculative shots
6. Document your strategic pivot reasoning in AGENT_LOG.md before resuming experiments

## Rule 7 — Force diversity after local optima
If you've made 3+ variations of the same technique (e.g. different shared memory tile sizes, different thread block dimensions, different radix widths), you are stuck in a local optimum. STOP tuning parameters and try a completely different algorithmic approach. The biggest GPU performance gains come from algorithmic changes and memory access pattern redesigns, not parameter sweeps.

**Special case — applying the same technique to each data type**: If you've applied the same optimization pattern (e.g. "fused delimiter scan + X conversion") to multiple data types sequentially, you are optimizing micro-benchmarks, NOT the mixed-type workload. STOP. The `csv_read_io` benchmark runs ALL types together — per-type micro-optimization has sharply diminishing returns because warp divergence, instruction cache pressure, and register spilling dominate in the mixed case. Switch to architecture-level optimizations (multi-pass reduction, memory bandwidth, host overhead, multi-stream pipelining).

## Rule 8 — Re-anchor every 5 experiments (prevents context rot)
Every 5th experiment, do ALL of the following — in long sessions, instructions loaded early fade from active attention as context grows:
1. Re-read `.claude/rules/discipline.md` from disk
2. Re-read `.claude/rules/experiment-safety.md` from disk
3. Re-read results.tsv AND AGENT_LOG.md end-to-end
4. Check `/memory` for discoveries from prior sessions and earlier in this session
5. Re-state your objective in one sentence (reminder: optimize `csv_read_io` multi-type benchmark)
6. Summarize which approaches worked, which failed, and why
7. Save any new insights to memory that haven't been persisted yet
8. Only THEN propose your next hypothesis

## Rule 9 — No scope creep
Do not "also optimize" a neighboring module. Do not refactor code outside the critical path. Do not add logging, profiling instrumentation, or utility functions unless directly required for the current experiment.

## Rule 10 — Write your hypothesis before writing code
Before implementing, state your hypothesis explicitly:
- What you're changing and why (grounded in GPU architecture or algorithm theory)
- What metric you expect to improve and by roughly how much
- What could go wrong

Each build cycle is expensive. This forces you to think before spending 10-30 minutes on a doomed idea.

## Rule 11 — Don't strip working code outside the hot path
If existing code handles edge cases, error paths, or special types that the benchmark doesn't exercise, leave it alone. Removing working functionality to simplify the benchmark path is metric gaming, not optimization. The benchmarks test common cases; the tests verify correctness for all cases.

## Rule 12 — Research head mindset before each experiment
Before each experiment, step back and think like a research head — not just an implementer:
- **Assess portfolio**: Which optimization directions have yielded gains? Which show diminishing returns?
- **Identify bottleneck**: For the `csv_read_io` multi-type benchmark, what is currently the #1 bottleneck? Target that.
- **Spin up researchers**: Spawn 2-3 focused researcher agents with specific research questions. Don't wait until the idea backlog is empty — proactive research produces better ideas.
- **Evaluate trajectory**: Rapid gains → keep pushing. Diminishing returns → pivot to new bottleneck.
- **Document strategy**: Write your strategic reasoning in AGENT_LOG.md so the direction is traceable.

## Rule 13 — Always log to AGENT_LOG.md
After every experiment (keep, discard, or crash), append a section to AGENT_LOG.md with: hypothesis, result, what worked, what didn't, and what was learned. This is append-only — never edit or delete previous entries. The journal is essential for the research head to assess trajectory.

## Rule 14 — Use memory to persist discoveries
After each significant finding (successful optimization, failed approach with clear reason, bottleneck identified, useful paper found), save it to `/memory`. Memory survives across sessions and context compaction. At the start of each session, check memory for prior discoveries — don't re-try approaches that already failed or re-discover known bottlenecks.

## Simplicity Criterion
- Equal perf + simpler code = **keep**
- 1% gain + 50 lines of hacks = **probably not worth it**
- 1% gain from deleting code = **definitely keep**
- Equal perf + much simpler code = **keep**

## When You Run Out of Ideas
This is NOT the time to make random changes to GPU kernel parameters. Instead:
1. Do more web searches — new papers, different search terms, adjacent GPU workloads
2. Re-read the source code from disk looking for bottlenecks you missed
3. Try combining two previous near-miss ideas that each showed partial improvement
4. Try a fundamentally different approach (e.g. if you've been optimizing memory access, try algorithmic complexity reduction instead)
5. Spawn the researcher agent to search for techniques in parallel

Random mutations waste build cycles. Research finds new strategies.

## Never Stop
Once the experiment loop begins, run indefinitely until manually interrupted. Never ask "should I continue?"

## Quality Over Quantity
Every rejected experiment wastes a full build-test-benchmark cycle (10-30 min). A well-reasoned hypothesis with high confidence is worth more than five speculative shots.

## Unlimited Budget
Cost is not a concern. Always use Opus 4.6 (1M context) at maximum effort. Do as many web searches as needed. Spawn researcher subagents freely. Context auto-compacts as needed. The only metric that matters is performance improvement.
