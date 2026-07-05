# Optimization guide

This document describes the performance decisions made by Regex IR's current
compiler and NVVM renderer, the limits that preserve regex semantics, and the
remaining optimization opportunities. It is a description of the checked-in
implementation, not a promise that every pattern takes the fastest path.

The core library compiles one pattern and one regex API operation at a time. It
does not own a CUDA context, a cuDF column, module caching, allocation, or kernel
launch policy. The benchmark and test adapters provide those integration
pieces. Consequently, performance has three distinct layers:

1. compile-time IR simplification and executor selection;
2. generated per-row device code; and
3. integration policy, including compilation caching, row scheduling, output
   allocation, and kernel launch geometry.

Optimizations must preserve leftmost-first matching, ordered alternation,
greedy/lazy priority, captures, zero-length progress, UTF-8 boundaries, and the
selected API's result shape. A transformation that is valid for boolean
existence is not automatically valid for find, extract, replace, or split.

## Optimization pipeline

The compiler uses the following pipeline:

```text
pattern
  -> ordered Thompson Automata IR
  -> operation-specialized Instruction IR
  -> graph and capture simplification
  -> executor analysis and deterministic construction
  -> operation-specific textual NVVM IR
  -> libNVVM PTX at -opt=3
  -> nvJitLink device LTO at -O3
  -> linked CUDA cubin
```

The optimizer is deliberately split between Instruction IR passes and NVVM
executor selection. Instruction IR stays inspectable and target-independent;
the NVVM renderer can then choose a code shape based on which results are
observable.

## Instruction IR passes

`regex_ir::optimize` applies these passes in order and verifies the graph before
and after the sequence:

| Pass | Current action | Performance purpose | Semantic constraint |
|:---|:---|:---|:---|
| unobserved capture removal | removes capture writes for boolean, find, count, and split; replacement retains only referenced captures; extract retains all captures | reduces stores, snapshots, live state, and register pressure | a capture is removed only when the selected API cannot observe it |
| epsilon-jump folding | bypasses linear instruction-free one-successor blocks, stopping at cycles | removes dispatch and empty control-flow plumbing | nullable cycles are not traversed indefinitely |
| literal fusion | combines a single-incoming linear chain of singleton predicates into one `match_literal`, up to 64 code points by default | exposes bounded peeks and literal-specialization opportunities | branch joins and cycles stop fusion |
| second epsilon fold | removes empty blocks exposed by literal fusion | shortens the final graph | same rule as the first fold |
| unreachable removal | compacts reachable blocks and rewrites dense block IDs | reduces generated code and analysis cost | reachability starts at the operation entry block |

These passes do not currently perform DFA minimization, common-subexpression
elimination, data-dependent planning, row-length analysis, or mandatory
substring extraction.

## Executor selection

The NVVM renderer analyzes the optimized IR and selects the first safe path in
the following conceptual order:

```text
non-boolean exact one-byte ASCII expression?
  yes -> direct byte finder
  no  -> deterministic construction succeeds?
           boolean -> existential or assertion-aware DFA
           span/global result without live captures -> ordered DFA
           extract with an unambiguous one-pass capture history -> tagged DFA
           otherwise -> ordered recursive Thompson fallback
```

The generated module comments identify the selected executor and, for a
deterministic machine, its state and alphabet-class counts. This is useful when
correlating a benchmark case with PTX, SASS, or an Nsight report.

### Boolean existential DFA

For `contains` and `matches`, capture history and greedy/lazy path priority do
not change the boolean result. The renderer therefore uses an existential
subset construction when it is representable within the resource limits.

For scanning `contains`, the start closure is injected into every next state.
This recognizes a match beginning at any input position in one forward pass;
it does not invoke a matcher separately at every candidate position. The state
encoding uses one acceptance bit and 15 state bits, permitting at most 32,767
encoded states.

For a beginning-anchored boolean expression, the renderer removes the
already-proven start assertion and builds a non-scanning machine. A
non-scanning machine rejects immediately on entry to its dead state rather
than reading the remainder of a known-failing row.

### Assertion-aware boolean DFA

Boolean zero-width assertions are included in deterministic epsilon closure.
The transition is indexed by both an input alphabet class and the relevant
position context, which can include:

- beginning or end of input;
- beginning or end of line;
- word or non-word boundary; and
- CRLF-sensitive extended-newline context.

This avoids the recursive fallback for boolean word boundaries and line
anchors. It increases the transition-table dimensions, so construction still
falls back when the state/table limit would be exceeded.

### Ordered DFA

Find, count, replacement, split, and capture-free span operations expose the
preferred match. Their subset states therefore preserve Thompson-thread order.
Each 16-bit transition stores 14 state bits, an acceptance bit, and a
`stop-before-accept` bit used to retain ordered alternation and greedy/lazy
priority. The state limit is 16,383.

The current ordered finder retries from successive logical-character
boundaries when a candidate fails. Unlike boolean `contains`, it does not fold
all possible start positions into a single streaming state, because it must
retain the winning start and path priority. Required-prefix filtering can
reduce retries only on the recursive path today. Candidate retry is therefore
an important remaining cost for some span-producing or global expressions.

### Tagged DFA

Capture extraction uses deterministic transitions only when analysis proves:

- no deterministic state has more than one matching consuming thread for an
  alphabet class;
- acceptance is terminal rather than mixed with continuing alternatives; and
- every route to a deterministic state has the same capture-update history.

When those conditions hold, each transition carries a small capture action
program and extraction runs without recursive Thompson calls. Ambiguous
capture histories fall back to preserve exact capture semantics.

### Recursive Thompson fallback

The fallback is the general correctness path for non-boolean internal
assertions and ambiguous capture histories. One recursive dispatcher executes
the cyclic Instruction IR block graph. It has a size-derived step limit to
bound nullable cycles.

The fallback retains several optimizations:

- an entry singleton or fused literal can supply a required first ASCII byte;
- candidate starts whose first byte does not match are skipped;
- `llvm.expect` marks a required-prefix hit as unlikely when branch hints are
  enabled;
- literal predicates receive specialized helpers;
- leaf helpers carry `alwaysinline`, `readonly`, `readnone`, and `nounwind`
  attributes where valid; and
- only capture slots live for the selected result are initialized or updated.

The filter is currently one required ASCII byte at the expression entry, not a
general mandatory-literal analysis. It is also specific to the recursive
fallback. Historical SASS and Nsight analysis found recursive call frames,
local-memory traffic, repeated decoding, and divergent backtracking to be the
dominant costs, so deterministic paths are preferred whenever semantics and
table bounds permit.

### Large boolean alternations

For a boolean graph with at least 80 blocks and an empty multi-way entry, the
renderer can compile top-level alternatives as separate recursively optimized
DFA functions. A short-circuit wrapper calls them in sequence.

This bounds state-product growth and allows one difficult branch to fall back
without forcing every branch into the same large machine. The tradeoff is
re-reading a row when multiple alternatives fail. It is intentionally limited
to boolean results, where alternative priority is not observable.

## Alphabet and transition representation

Deterministic construction partitions Unicode into equivalence classes: two
code points share a class when every consuming NFA predicate treats them the
same. The machine therefore stores `states x classes`, not `states x 1,114,112`,
transitions.

The generated representation has:

- a 256-entry constant-memory table for byte-to-class mapping;
- a generated comparison/select chain for non-ASCII Unicode intervals;
- 16-bit transition entries with state and control flags packed together; and
- a hard cap of 4,194,304 transition items during construction.

Transition tables up to 32 KiB use NVVM constant address space. Larger tables
use read-only global storage so multiple tables cannot exhaust the device's
64-KiB constant segment. The renderer emits ordinary loads and does not force a
cache operator; PTX generation and device linking choose the cache policy.

ASCII input avoids full UTF-8 decoding. Non-ASCII input decodes a code point,
computes its byte width, then classifies it. Even the ASCII loop still performs
a class-table lookup and a state-table lookup per consumed byte. Profiles of
complex deterministic expressions therefore point to classification and
transition instruction count as a remaining optimization target.

## Literal and anchor specialization

Several local analyses avoid general regex machinery:

- A complete, unanchored, capture-free, one-byte ASCII expression used by a
  non-boolean operation receives a direct raw-byte finder. ASCII bytes cannot
  occur inside a valid UTF-8 continuation sequence, so logical-boundary
  semantics are preserved.
- Beginning anchors on boolean expressions become non-scanning control flow.
- Non-scanning deterministic machines stop at their dead state.
- Fused literals reduce helper and cursor operations in the fallback.
- A replacement reference to a capture proven to cover the whole match reuses
  the match span instead of allocating and maintaining a separate capture.

There is not yet an exact multi-byte literal executor. A two-byte or longer
literal normally uses a deterministic table or fallback helpers rather than a
wide comparison, KMP, Two-Way, or Boyer-Moore-family search.

## API specialization

Only the selected API is emitted into a module. There is no runtime regex
opcode switch and no generic result union.

| API | Per-row generated behavior | Important cost consideration |
|:---|:---|:---|
| `contains` | returns on the first accepting state | existential scanning DFA can cover all starts in one pass |
| `matches` | starts at byte zero and requires the operation's anchor/end semantics | dead-state rejection avoids a known-useless tail scan |
| `find` | stores the first preferred begin/end pair | ordered candidate retries may dominate on long near-misses |
| `count` | enumerates non-overlapping matches and handles zero-length progress | calls the selected finder repeatedly |
| `extract` | writes the whole-match and numbered capture spans | tagged DFA is used only for capture-safe one-pass graphs |
| `replace` | copies unmatched ranges and a compile-time replacement template | null output sizes; non-null output materializes; referenced captures remain live |
| `split` | counts or writes field spans around non-overlapping matches | null output sizes; non-null output writes spans |

Replacement constants live in constant storage, and range copies use
`llvm.memcpy`. Replace and split expose sizing and emission through the same
device function by accepting a null output pointer.

At the cuDF-column integration layer, variable-size replace and split use a
sizing kernel, a device prefix scan/allocation, and an emission kernel. The
current emission pass matches the rows again. This avoids pessimistic
over-allocation but can nearly double matcher work when matching is more
expensive than output construction. Persisting compressed match spans is a
possible time-for-memory tradeoff.

## Compiler and linker optimization

The core API returns textual NVVM IR and does not invoke CUDA tools itself. The
test and benchmark integrations use:

- libNVVM verification for the selected compute architecture;
- libNVVM compilation with `-opt=3`;
- an NVCC-built kernel-wrapper LTO fatbin;
- nvJitLink with `-lto -O3`; and
- module loading of the linked cubin.

Symbol prefixes isolate generated internals, and the public execute-function
name lets the stable wrapper call the specialized matcher. Production
integrations should cache linked cubins by at least pattern, operation,
compile/codegen options, GPU architecture, and CUDA toolkit version. JIT
specialization targets repeated workloads; uncached one-shot compilation is
not expected to beat a precompiled interpreter's setup latency.

## Current launch and column policy

The GPU benchmark adapters consume cuDF STRING columns and produce owning cuDF
columns. They map one CUDA thread to one input row and use 256-thread blocks,
chosen from register/occupancy profiling. Consecutive rows remain in their
original order and are addressed through the cuDF offsets column.

The current integration does not:

- sort or bucket rows by length;
- transpose strings into a pivoted layout;
- assign multiple lanes to a long row;
- use a persistent work queue for long-tail rows;
- collect input-byte or match-selectivity statistics for executor selection;
- vector-load input in the general matcher; or
- combine several regex programs into one multi-pattern automaton.

These omissions are policy choices and future opportunities, not claims that
the techniques are universally unhelpful. Their preprocessing and temporary
storage must be included in end-to-end measurements.

## Profile-guided findings

The detailed measurements are retained in the README's profile-guided
optimization section. The conclusions that currently guide the code are:

- recursive Thompson execution was limited by call-frame/local-memory traffic,
  repeated UTF-8 work, low active-lane efficiency, and register pressure;
- ordered and tagged determinization removed that traffic and substantially
  improved occupancy and active lanes;
- direct one-byte global operations were previously paying full ordered-DFA
  overhead; the byte finder reduced instruction count by roughly eightfold in
  the profiled split kernels;
- non-scanning anchored failures were reading dead states to end-of-row; early
  dead-state rejection removed that work;
- current direct-byte kernels are memory-throughput limited; and
- profiled complex complete-corpus DFA kernels are compute/instruction limited,
  with low DRAM utilization, high branch efficiency, no spills, and remaining
  costs in classification, transitions, and divergent row completion.

Branch-free code is not an objective by itself. A specialized DFA still needs
loop, end-of-input, ASCII/UTF-8, and acceptance control flow. The useful goal is
to remove input-dependent regex dispatch and backtracking while keeping the
remaining branches uniform or strongly biased.

## Relation to the Sitaridi and Ross paper

[GPU-accelerated string matching for database
applications](https://doi.org/10.1007/s00778-015-0409-y) studies exact
single- and multi-pattern database search on C2070 and K40 GPUs. The authors'
[thesis chapter](https://www.cs.columbia.edu/~eva/gpu_thesis.pdf) gives the
full algorithms and evaluation. Its central lessons are still relevant:

- independent threads can create excessive L2 footprint and uncoalesced
  accesses;
- similar-length grouping reduces idle lanes at row completion;
- splitting work into stages can compact unfinished rows and reduce
  divergence;
- segmenting or pivoting input can improve cache-line reuse and coalescing;
- wide loads reduce load instructions and cache traffic;
- regular KMP access is more robust than data-dependent skipping on
  adversarial inputs; and
- algorithm, thread-group size, layout, selectivity, and device generation
  must be tuned together.

The paper is not a direct design for this project. Most of its GPU evaluation
is substring/SQL-LIKE matching over preprocessed layouts, while Regex IR must
preserve general regex priority, assertions, captures, and materializing API
semantics on ordinary cuDF columns. Its pivoting cost can be amortized over a
database column and repeated queries; it may lose end-to-end on a one-shot
regex call. Its KMP result applies directly to literal-only or extracted
literal filters, not as a replacement for a general tagged or ordered regex
automaton.

## Assessment of common optimization proposals

| Proposal | Current status | Assessment |
|:---|:---|:---|
| JIT/operation specialization | implemented | core design; warm execution benefits, while linked cubins should be cached |
| profile occupancy, memory use, and divergence | implemented as a development practice | current complex DFA cases are more instruction/lane limited than DRAM limited; direct-byte cases are bandwidth limited |
| Aho-Corasick | not implemented | useful for many exact literals or a bank of extracted literals, not a general regex replacement; dense transition storage, not automaton state count alone, is the main GPU memory risk |
| cheap filter then full regex | partially implemented | one required entry ASCII byte exists only in the fallback; broader mandatory-literal and selectivity-aware filtering is a high-value gap |
| group strings by length | not implemented | promising for high length variance and repeated scans, but binning/permutation/scatter cost must be amortized |
| pivot strings | not implemented | potentially useful for stable, repeatedly scanned fixed/coarsely bucketed columns; too costly as an unconditional cuDF API step |
| staged unfinished-row compaction | not implemented | promising for selective contains and highly skewed rows, but extra kernels/global traffic can outweigh divergence savings |

Aho-Corasick is specifically a multi-literal trie plus failure transitions. It
does not make arbitrary regex matching a DFA, and its number of states is
normally linear in the total literal length. A full regex DFA can suffer
subset-state explosion; an Aho-Corasick implementation more commonly suffers
from a large alphabet-by-state transition representation. The two concerns
should not be conflated.

## Prioritized opportunities

The following work is ordered by likely value for the current implementation.
Each item needs end-to-end benchmarks over API, row count, row width, length
variance, hit rate, regex complexity, and reuse count.

### 1. Add a literal planner

Recognize exact literal expressions and mandatory literals during IR analysis.
Useful specializations include:

- anchored exact compare for `matches`;
- wide fixed-length compares for short ASCII literals;
- KMP or another regular linear search for longer literal-only expressions;
- direct occurrence enumeration for count, replace, and split; and
- a required first-byte, first-byte-set, prefix, or longer-literal filter before
  an expensive fallback or ordered candidate attempt.

For a mandatory literal at a variable regex offset, a row-level filter can
reject misses but cannot always infer the match start. For a fixed prefix it
can identify candidate starts directly. That distinction belongs in the
compile-time analysis.

### 2. Make filtering selectivity-aware

A cheap filter is not automatically cheap: a separate pass doubles input
traffic on rows that survive. Provide at least two plans:

- a fused per-row filter that immediately verifies a hit; and
- for very selective filters and expensive verifiers, a bitmap/row-ID pass
  followed by compaction and a verification kernel.

Choose between them using sampled literal frequency, row widths, regex cost,
and expected cubin reuse. Measure preprocessing, compaction, and result scatter
inside the end-to-end path.

### 3. Remove ordered candidate restarts where safe

Boolean contains already carries the start closure through one streaming DFA.
Span/global operations still retry candidates. Investigate a prioritized
streaming or tagged-search automaton that carries the earliest live start and
only the capture state required by the selected API. This can improve the
algorithmic shape of long near-misses, but it must reproduce leftmost-first and
greedy/lazy results exactly.

### 4. Reuse match work in materializing APIs

Replacement and split currently match during both sizing and emission. For
selective or complex expressions, store compact spans during sizing and consume
them during emission. Select between rematching and span scratch based on match
density and scratch size; the direct one-byte path may still be faster when it
rematches than when it writes and rereads a large span array.

### 5. Reduce deterministic-loop instruction count

Profile and compare:

- inline class comparisons or bitmaps for very small ASCII alphabets;
- specialized all-ASCII loops when column metadata or sampling justifies it;
- processing 4, 8, or 16 input bytes per load for literal scans;
- loop unrolling only when it does not worsen register pressure;
- DFA minimization for existential boolean machines;
- hot-state numbering and transition-row layout; and
- smaller transition encodings when state/class counts allow them.

The complete-corpus profiles indicate this is more likely to help complex DFA
cases than cache-policy forcing.

### 6. Schedule skewed rows more deliberately

Start with coarse length bins rather than a full sort. Build a row-index
permutation from the already-available offsets, launch a small number of bins,
and write results to original row indices. Enable it only when length variance,
mean width, and reuse predict that reduced idle lanes repay histogram,
permutation, and launch costs.

For extreme long tails, compare bins with a persistent queue or multi-lane
segmentation of long rows. Segment boundaries need overlap or carried automaton
state; ordered captures make this harder than exact KMP search.

### 7. Treat pivoting as a cached column representation

Do not transpose every cuDF input unconditionally. A pivoted or tiled view is
most plausible when the same stable column is scanned by many patterns. Cache
the transformed representation, use coarse length classes to limit padding,
and include transform construction and cache lifetime in the API design.

### 8. Add multi-pattern filtering only for a real workload

If consumers need many regex predicates over the same column, an
Aho-Corasick filter over exact patterns or extracted mandatory literals can
produce candidate `(row, pattern)` pairs. Specialized regex functions can then
verify only those candidates. Keep the verifier separate from the literal
automaton to avoid the state product of a combined full-regex DFA.

## Measurement checklist

For every proposed change, compare the current executor and candidate plans
across:

- API: contains, matches, find, count, extract, replace, and split;
- row count and total input bytes;
- mean, maximum, and coefficient of variation of row length;
- match and filter selectivity, including adversarial near-misses;
- ASCII and non-ASCII proportions;
- regex states, alphabet classes, assertions, captures, and output density;
- cold compilation, first launch, and warm execution;
- one-shot versus repeated use, including preprocessing amortization; and
- complete output-column construction, not matcher-kernel time alone.

Nsight Compute should track active threads per warp, branch efficiency,
eligible/active warps, achieved occupancy, registers, local-memory traffic,
long-scoreboard stalls, instruction count, L1/L2 hit rates, and DRAM throughput.
Nsight Systems should confirm whether time belongs to compilation, module load,
allocation/scan, launches, or device execution. Profiler replay changes
absolute latency, so presentation and README performance tables should continue
to use normal NVBench timings and use profiler data only to explain them.
