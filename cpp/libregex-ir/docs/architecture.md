# Architecture

## Scope

The core library compiles IR and provides a correctness-first NVVM IR text
renderer. A downstream project still owns compiler
invocation, input/output storage, module loading, kernel launch policy, and
allocation. The CPU executor under tests is not installed. The optional GPU
benchmark owns those runtime pieces only for its cuDF comparison workload.

## Stages

1. The parser validates and normalizes the supported pattern language.
2. Ordered Thompson construction creates Automata IR with consuming, branch, assertion, capture, jump, and accept states.
3. Lowering creates one typed structured-control-flow block per Automata state.
4. Optimization specializes captures, bypasses empty jumps, fuses literal chains, and removes unreachable blocks.
5. Both IR levels are verified before being returned.
6. The NVVM generator partitions Unicode into predicate-equivalent alphabet
   classes. Boolean programs use existential deterministic states because
   match priority cannot change a boolean result. Span/global programs retain
   ordered states, and capture graphs proven to have one consuming thread per
   class and terminal acceptance receive tagged transitions. Other programs
   retain ordered graph lowering.
7. The selected executor is rendered directly to NVVM IR without a runtime
   regex opcode interpreter. Boolean, find, count, extract, replacement, and
   split each receive a distinct public device ABI and operation loop.

The Automata representation is Thompson rather than the initially considered Glushkov position NFA. It has linear construction size and directly preserves capture tags, embedded assertions, empty paths, and greedy/lazy branch order. Instruction lowering and optimization remove unnecessary epsilon plumbing from the consumer-facing form.

## Safety and determinism

- IDs are stable storage indices.
- Printers are deterministic.
- Pattern, nesting, repeat, state, transition, and capture limits are configurable.
- Failure returns diagnostics and no partial IR.
- Each optimization runs between verifier checks.
- Unsupported dialect constructs are rejected explicitly.

## GPU orientation

Instruction IR expresses bounded peeks, character predicates, literal windows,
cursor advancement, branches, assertions, capture writes, and acceptance.
Static metrics expose block, branch, predicate, read, capture-write, and literal
sizes. An ordered subset construction retains alternation and greedy/lazy
priority for result shapes that observe a match span, including a
stop-before-accept transition flag. Boolean machines reclaim that unused flag
bit for a 15-bit state index. Large boolean alternations can become independent
DFA functions behind a short-circuit wrapper, and tables too large for the
device constant segment use read-only global storage. Boolean zero-width
assertions are determinized by indexing epsilon closures with the relevant
begin/end, word-boundary, line-boundary, and CRLF predicate bits. Tagged
transitions record capture boundaries when their histories are provably
unambiguous. Non-boolean internal assertions and ambiguous capture histories
use the recursive Thompson fallback, which retains required-prefix filtering,
direct ASCII-literal lowering, optimizer attributes, and branch hints. All
paths use ordinary input loads so the CUDA compiler selects cache behavior.
Every module contains only
its selected consumer: scalar counting, capture extraction, replacement
sizing/emission, or split sizing/span emission. Capture writes survive only
when the result observes them.

Actual occupancy, register pressure, divergence, and bandwidth require
representative hardware. The optional `regex-ir-gpu-benchmark` reports
warm/cold latency and throughput for the built-in NVVM path beside cuDF's
contains, count, extract, replace, and split APIs. The separate
`regex-ir-corpus-benchmark` covers OpenResty, Rust Leipzig, Boost/GCC, and
mariomka expression inventories over complete, checksum-pinned source corpora.
Both paths consume owning cuDF STRING columns. Their timed result contract is
also identical: BOOL8 for contains, INT32 for count, a table of STRING captures
for extract, STRING for replace, and LIST<STRING> for split. Regex IR's adapter
constructs those owning objects inside the timed region and recursively
compares each setup result with cuDF before sampling. Neither driver makes a
hardware-independent performance claim.
