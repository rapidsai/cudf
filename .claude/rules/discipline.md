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

## Rule 6 — Stall detection and deep research phase
If 3+ experiments show no improvement across ANY primary benchmark (including within-noise-floor results, not just `discard`/`crash`), you are stalled. STOP experimenting and enter **deep research phase**:
1. Re-read the source code from disk (not from memory — your cached understanding may be wrong after many edits)
2. Re-read results.tsv AND AGENT_LOG.md end-to-end
3. Read NVTX stage profiling — reassess which stage is the actual bottleneck
4. Spawn **2-3 deep researcher agents** with the full experiment history. They must find a fundamentally new direction.
5. Require a **high-confidence idea** (backed by a paper, clear architectural insight, or profiling data) before starting the next experiment. Don't spend another build cycle on a hunch.

## Rule 7 — Force diversity after local optima
If you've made 3+ variations of the same technique (e.g. different shared memory tile sizes, different thread block dimensions, different radix widths), you are stuck in a local optimum. STOP tuning parameters and try a completely different algorithmic approach. The biggest GPU performance gains come from algorithmic changes and memory access pattern redesigns, not parameter sweeps.

## Rule 8 — Research head: assess before every hypothesis
Before forming each hypothesis, act as research head:
1. Read the latest `nvtx_stages.txt` — which stage dominates?
2. Read the last entries of `AGENT_LOG.md` — what patterns are emerging?
3. Spawn 1-2 small focused researcher agents targeting the specific bottleneck.
4. Only then form a research-backed hypothesis.
Each experiment should be informed by targeted research, not just intuition from the previous result.

## Rule 9 — Re-anchor every 5 experiments (prevents context rot)
Every 5th experiment, do ALL of the following — in long sessions, instructions loaded early fade from active attention as context grows:
1. Re-read `.claude/rules/discipline.md` from disk
2. Re-read `.claude/rules/experiment-safety.md` from disk
3. Re-read results.tsv and AGENT_LOG.md end-to-end
4. Check `/memory` for discoveries from prior sessions and earlier in this session
5. Re-state your objective in one sentence
6. Summarize which approaches worked, which failed, and why
7. Save any new insights to memory that haven't been persisted yet
8. Only THEN propose your next hypothesis

## Rule 10 — No scope creep
Do not "also optimize" a neighboring module. Do not refactor code outside the critical path. Do not add logging, profiling instrumentation, or utility functions unless directly required for the current experiment.

## Rule 11 — Write your hypothesis before writing code
Before implementing, state your hypothesis explicitly:
- What you're changing and why (grounded in GPU architecture or algorithm theory)
- What metric you expect to improve and by roughly how much
- What could go wrong

Each build cycle is expensive. This forces you to think before spending 10-30 minutes on a doomed idea.

## Rule 12 — Don't strip working code outside the hot path
If existing code handles edge cases, error paths, or special types that the benchmark doesn't exercise, leave it alone. Removing working functionality to simplify the benchmark path is metric gaming, not optimization. The benchmarks test common cases; the tests verify correctness for all cases.

## Rule 13 — Use memory to persist discoveries
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

## Budget
Always use Opus 4.6 (1M context) at maximum effort. Do as many web searches as needed. Spawn researcher subagents freely. Context auto-compacts as needed. The only metric that matters is performance improvement.
