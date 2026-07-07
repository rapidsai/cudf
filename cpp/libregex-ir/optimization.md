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
  -> executor analysis and DFA/position construction
  -> operation-specific textual NVVM IR
  -> libNVVM LTO IR at -opt=3 -gen-lto
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
exact ASCII expression?
  yes -> direct compare, byte finder, packed finder, or long-literal scan
  no  -> assertion-free, non-nullable boolean graph with at most 64 positions?
           yes -> compare a bit-parallel Glushkov plan with the existential DFA
           no  -> continue to deterministic construction
         deterministic construction succeeds?
           boolean -> existential or assertion-aware DFA
           span/global result without live captures -> ordered DFA
           extract with an unambiguous one-pass capture history -> tagged DFA
           otherwise -> ordered recursive Thompson fallback
```

The generated module comments identify the selected executor and, for a
deterministic machine, its state and alphabet-class counts. Glushkov modules
report position, alphabet-class, shift, and exception counts. This is useful
when correlating a benchmark case with PTX, SASS, or an Nsight report.

### Bit-parallel Glushkov NFA

Boolean `contains` can use a Glushkov position automaton when the optimized
graph has no assertions, is non-nullable, and has at most 64 consuming
positions. The canonical public IR remains ordered Thompson IR; the Glushkov
machine is a private NVVM code-generation plan derived with iterative epsilon
closure. It therefore adds no public dialect and cannot introduce recursive
compiler traversal.

One `uint64_t` is the complete per-row regex state. Bit `i` means that position
`i` matched the preceding logical character. For the next character `c`, the
generated loop computes:

```text
next = reach(c) & (first | follow(active))
```

`contains` injects `first` on every character, so all possible starts are
processed in one forward pass. `matches` injects it only at byte zero and tests
acceptance at end of input. Accepting positions are another compile-time mask.
Because these APIs expose only existence, no two-phase rescan, match-start
tracking, alternative-priority kill, or capture history is needed. Those
features would be required before a position automaton could implement find,
extract, replace, or split semantics.

Follow edges with the same positive position delta are combined into at most
eight masked shifts. Back edges, zero-delta edges, and less frequent positive
deltas become explicit exception masks. Alphabet-equivalent input characters
map to precomputed reach masks. Small reach sets are emitted as immediate
selects; larger sets use constant storage. The JIT embeds this read-only data
in the generated module, so the cuDF proposal's cooperative shared-memory copy
was not adopted: for the accepted plans there is no large generic program
object to stage, and adding launch-time shared-memory policy would violate the
core library's integration boundary.

Glushkov is selected only when profiling predicts a material advantage:

- the competing DFA transition table exceeds 32 KiB and the position graph has
  at most five exception source positions; or
- the graph has at least 32 positions, exactly one forward shift, and no
  exceptions.

Otherwise the existing specialized literal or DFA path remains selected.
Assertions, nullability, more than 64 positions, and result shapes that expose
spans, priority, or captures automatically use the established paths. This is
a performance fallback, not a semantic restriction or a user-facing option.

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

Acceptance is an ordered state marker, not an unordered property of the DFA
subset. When an earlier Thompson thread can continue past a match from a later
thread, the later acceptance remains deferred in the next state. A transition
updates the accepted end position only when that transition discovers a new
higher-priority acceptance; merely carrying a deferred acceptance must not
overwrite it. If none of the earlier threads can consume the next character,
`stop-before-accept` returns the deferred match without consuming that
character. This distinction is required for lazy repetition, ambiguous
alternation, and nullable branches in find, count, replacement, and split.

The repair was checked by replaying all 43,358 span-producing cases from the
fixed-seed 45-minute GPU differential campaign across two RTX A6000 devices;
the replay produced zero CPU-oracle mismatches.

The ordered finder normally retries from successive logical-character
boundaries when a candidate fails. Unlike boolean `contains`, it cannot fold
all possible start positions into an existential state because it must retain
the winning start and path priority. Two compile-time proofs reduce that cost:

- a non-nullable machine with at most 16 possible starting bytes in at most two
  ASCII ranges receives an inline start-range predicate before initialization;
  and
- if every initial consuming class enters the same non-accepting self-loop
  state, those classes loop in that state, and no other state can re-enter it,
  a failed prefix run is skipped as one unit.

The latter rule preserves leftmost-first behavior because every later start
inside the skipped run reaches the identical state at the failure position.
Machines that accept the prefix, have stop-before transitions, multiple prefix
states, or another incoming edge do not use the acceleration.

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
cache operator; libNVVM and device linking choose the cache policy.

ASCII input avoids full UTF-8 decoding. Non-ASCII input decodes a code point,
computes its byte width, then classifies it. When the 256-byte alphabet differs
from its most frequent class in no more than two contiguous intervals, the
renderer emits inline range comparisons and selects instead of a byte-class
table load. More fragmented alphabets retain the constant table. Every DFA
still performs a transition lookup per consumed character, so transition-loop
instruction count remains a target for larger machines.

## Literal and anchor specialization

Several local analyses avoid general regex machinery:

- A complete linear exact-ASCII expression is recognized after IR
  optimization. One-byte global operations receive a direct raw-byte finder;
  multi-byte span/global operations use a first-byte guard and packed 8/4/2/1
  byte comparisons. ASCII bytes cannot occur inside a valid UTF-8 continuation
  sequence, so logical-boundary semantics are preserved.
- Anchored boolean exact matches compare the fixed literal directly. Scanning
  boolean literals of 2–15 bytes use the optimized DFA because PGO found it
  superior on early high-selectivity matches. Literals of at least 16 bytes
  scan eight possible first bytes per load with a byte-equality mask and invoke
  the packed verifier only for candidates.
- Beginning anchors on boolean expressions become non-scanning control flow.
- Non-scanning deterministic machines stop at their dead state.
- Fused literals reduce helper and cursor operations in the fallback.
- A replacement reference to a capture proven to cover the whole match reuses
  the match span instead of allocating and maintaining a separate capture.

The exact planner does not yet generate a failure-function KMP or Two-Way
search, nor does it extract mandatory literals at variable offsets from a
general expression.

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
- libNVVM compilation with `-opt=3 -gen-lto`;
- an NVCC-built kernel-wrapper LTO fatbin;
- nvJitLink with `-lto -O3`; and
- module loading of the linked cubin.

Every Regex IR benchmark state also reports an uncached JIT-ready interval.
It starts at the source regex, disables nvJitLink's cache, and stops after the
linked module is loaded and every required kernel function is resolved. Input
construction, output allocation, and the first launch are excluded.

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
- a bounded Glushkov position plan removes large boolean transition-table
  loads when the follow graph is sparse, while exception-heavy graphs remain
  faster as DFAs;
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
| bit-parallel Glushkov NFA | implemented for gated boolean plans | valuable for sparse follow graphs and DFA state growth; the forced all-pattern experiment regressed exception-heavy cases, so it is not the canonical IR or a universal replacement |
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

## 2026-07-05 highest-value-opportunity campaign

All eight opportunities below were exercised on an RTX A6000 with CUDA 13.2,
cuDF 26.08, Release compilation, libNVVM `-opt=3`, and nvJitLink `-lto -O3`.
Normal timings came from NVBench; Nsight Compute full-set replay was used only
for diagnosis. The primary gate was the existing 2,097,152-row,
`StringBytes=128` API matrix. Output allocation and owning cuDF-column
construction remained inside the timed region.

The final accepted code changed the geometric-mean latency of that gate as
follows. Small sub-millisecond contains differences have higher relative noise;
count and materializing wins are substantially larger than the measured noise.

| API | Cases | Before (ms) | After (ms) | Speedup |
|:---|---:|---:|---:|---:|
| contains | 22 | 0.747 | 0.708 | 1.056x |
| count | 7 | 4.347 | 3.381 | 1.286x |
| extract | 3 | 7.914 | 7.922 | 0.999x |
| replace | 14 | 16.818 | 12.089 | 1.391x |
| split | 7 | 22.800 | 17.030 | 1.339x |

The largest individual accepted improvement was `[a-z]+Z`: count fell from
6.828 to 3.689 ms, plain replacement from 20.493 to 11.121 ms, backreference
replacement from 19.797 to 11.189 ms, and split from 19.895 to 12.930 ms.

### Opportunity outcomes

| Opportunity | Experiment | Outcome |
|:---|:---|:---|
| literal planner | exact-ASCII graph recognition, packed comparisons, 8-byte first-byte masks, and short-literal DFA A/B | accepted; short scanning literals use the DFA, fixed/non-boolean and long literals retain direct paths |
| selectivity-aware filtering | contains sweep at 1%, 5%, 10%, 50%, and 100% hits | fused plan retained; it already sustained about 156–253 GB/s on low-to-moderate selectivity, leaving no one-shot margin for a second scan plus compaction/scatter |
| ordered restart removal | sparse start ranges and a proved self-loop prefix-run skip | accepted; count improved 16–46% across the seven transform expressions |
| materialization reuse | temporary one-pass staged-row replacement versus current exact-size rematch | rejected; staged allocation and compaction increased latency 26–157% |
| deterministic-loop reduction | inline byte-class ranges for alphabets with at most two exceptional intervals | accepted; several ordered count cases improved another 4–27%, with boolean cases neutral within noise |
| length scheduling | temporary physically length-sorted copy of the existing normal-width corpus | matcher-only upper bound was 7–16% faster, but sorting/gather/scatter was excluded; not enabled for one-shot APIs |
| pivoted representation | temporary fixed-width 2,097,152×128 literal probe | cached scan was 4.45x faster, but the 21.05 ms pivot plus 0.68 ms scan was 7.1x slower than the 3.03 ms row-major one-shot scan; keep as a cached-column design only |
| multi-pattern filter | temporary four-literal corpus producing four predicate arrays | one fused scan took 2.91 ms versus 10.30 ms for four scans, a 3.54x win; no single-pattern ABI change until a real multi-pattern consumer exists |

The complete 58-case large-corpus Regex IR gate also improved: geometric mean
latency fell from 0.366 to 0.340 ms, a 1.076x speedup. Boost cases 1, 2, and 6,
Leipzig cases 1 and 11, and OpenResty cases 1 and 18 improved by 23–36%.
The only apparent slower state in the first sweep was a 0.169 ms OpenResty
case; a longer 0.5-second rerun measured 0.170 ms, within 0.6% of its baseline.

### Temporary workload definitions

The temporary probes were removed after measurement. They are described here
so the conclusions are reproducible without mistaking them for shipped
benchmarks:

- **staged replacement:** 262,144 variable-length rows capped at 128 bytes,
  the existing transform patterns 0, 2, 5, and 6, and a per-row temporary
  stride of `2 * StringBytes`. The generated replacement ran once into the
  temporary, sizes were scanned, then a second kernel compacted bytes into the
  owning cuDF STRING child. Existing rematch times were 2.17–3.00 ms; staged
  times were 2.74–5.73 ms.
- **length ordering:** the existing seeded normal row-length distribution at
  2,097,152 rows, widths 128 and 256, and count patterns 2, 5, and 6. Rows were
  physically reordered by length before upload. This deliberately measured a
  cached-layout upper bound; permutation construction and output restoration
  were not timed.
- **pivot:** fixed 128-byte rows, 50% containing `0987 5W43` at byte 59,
  15 median event-timed samples after two warm-ups. Both layouts produced the
  same boolean output. The one-shot result includes the row-major-to-pivot
  transpose.
- **multi-pattern:** fixed 128-byte rows with `error:`, `https://`, `@host.`,
  `192.168.`, or no literal in a five-row cycle. Four separate specialized
  kernels and one fused kernel both wrote four boolean arrays; 15 median
  samples followed two warm-ups.

Two Nsight Systems CUDA traces completed the benchmark process but exceeded
120- and 180-second limits while finalizing their reports in this environment,
so no partial `.nsys-rep` is treated as evidence. Nsight Compute full profiles
completed normally. The optimized `[a-z]+Z` count kernel used 78 registers per
thread, achieved 43.37% warp occupancy, executed 1.287 billion SASS
instructions, and had 89.69% uniform branch targets. Its low 6.29 active-thread
ratio explains the cached length-ordering headroom, but the end-to-end
reordering cost still prevents enabling it unconditionally.

## 2026-07-07 Glushkov experiment

The experiment used the design and implementation discussion in
[rapidsai/cudf#21936](https://github.com/rapidsai/cudf/pull/21936) as a reference,
then adapted the idea to Regex IR's JIT boundary. The cuDF proposal stores a
general position program and cooperatively caches it in shared memory. Regex IR
instead specializes follow and reach masks into NVVM constants and immediate
operations for one pattern. It also limits the position engine to boolean
existence, where leftmost-first start and capture reconstruction are
unobservable.

The first forced-plan sweep ran all 74 complete-corpus Regex IR cases. It
validated every result against the existing cuDF setup oracle and measured a
1.021x geometric-mean speedup, but it also exposed 11 regressions above 2%.
Exception-heavy bounded graphs were the wrong fit: Boost/GCC's names-near-river
case fell to 0.647x, the quoted-string case to 0.754x, and IPv4 to 0.912x.
Keeping a universally selected Glushkov path would therefore have failed the
experiment despite the positive average.

The retained cost gate was evaluated with baseline/branch ABBA ordering because
the available account could not lock RTX A6000 clocks. Twenty-five or more
NVBench samples and a 0.5-second minimum sampling interval were used per state.
Representative accepted results were:

| Corpus case | Existing DFA (ms) | Glushkov (ms) | Speedup | JIT-ready effect |
|:---|---:|---:|---:|:---|
| OpenResty 23, `[a-q][^u-z]{13}x` | 0.497 | 0.358 | 1.388x | about 92–103 ms to 48–50 ms |
| Rust Leipzig 6, same expression | 0.394 | 0.288 | 1.367x | about 91–99 ms to 48–55 ms |
| OpenResty 8, 57-position fixed alternatives | 0.564 | 0.457 | 1.234x | neutral, because its DFA was already small |
| OpenResty 7, same expression on an early-result corpus | 0.197 | 0.198 | 0.999x | neutral |
| mariomka IPv4 negative control | 0.266 | 0.266 | 1.000x | DFA retained |

The bounded expression previously produced 16,385 DFA states and a read-only
global transition table. Glushkov represents it with 15 position bits, one
shift, no exception edges, and no transition table. A full Nsight Compute
replay of OpenResty case 23 explained the normal-timing gain:

| Metric | DFA | Glushkov |
|:---|---:|---:|
| profiled kernel duration | 65.92 us | 46.37 us |
| global-load instructions | 274,118 | 138,083 |
| long-scoreboard cycles per issued instruction | 11.40 | 5.10 |
| eligible warps per scheduler | 0.157 | 0.258 |
| registers per thread | 40 | 40 |
| achieved occupancy | 24.07% | 24.48% |
| branch efficiency | 96.73% | 96.73% |

Executed instructions increased from 4.72 million to 4.99 million; the win is
not fewer arithmetic instructions, but eliminating the large data-dependent
transition-table load. The profile also found an LTO device-call frame in the
benchmark wrapper. Marking the Glushkov public entry `alwaysinline` was tested
and rejected: controlled normal timings were neutral, so the attribute was not
retained without evidence that it changes linked code profitably.

Temporary diagnostic output used to record position, class, shift, exception,
and DFA-state counts was removed after the gate was chosen. Baseline
executables, normal NVBench JSON, and full `.ncu-rep` files were kept outside
the source tree during the campaign; no profiler-replay duration is used in
presentation tables.

## Prioritized opportunities

The following work is ordered by likely value for the current implementation.
Each item needs end-to-end benchmarks over API, row count, row width, length
variance, hit rate, regex complexity, and reuse count.

### 1. Extend the literal planner

Exact ASCII expressions and sparse first-byte ranges are implemented. The
remaining work is mandatory-literal extraction from general expressions and a
regular long-literal algorithm for adversarial repeated prefixes. Useful
specializations still include:

- anchored exact compare for `matches`;
- KMP or another regular linear search for longer literal-only expressions;
- a longer mandatory-literal filter before an expensive fallback; and
- selectivity estimates that can choose the long-literal path without a fixed
  size threshold.

For a mandatory literal at a variable regex offset, a row-level filter can
reject misses but cannot always infer the match start. For a fixed prefix it
can identify candidate starts directly. That distinction belongs in the
compile-time analysis.

### 2. Make cross-row filtering selectivity-aware

A cheap filter is not automatically cheap: a separate pass doubles input
traffic on rows that survive. The fused plan is implemented and won the
existing benchmarks. A future multi-kernel plan would need:

- a fused per-row filter that immediately verifies a hit; and
- for very selective filters and expensive verifiers, a bitmap/row-ID pass
  followed by compaction and a verification kernel.

Choose between them using sampled literal frequency, row widths, regex cost,
and expected cubin reuse. Measure preprocessing, compaction, and result scatter
inside the end-to-end path.

### 3. Generalize ordered restart removal

Sparse starts and self-looping prefix runs now avoid many retries. Other
span/global machines still retry candidates. A prioritized streaming or
tagged-search automaton could carry the earliest live start and only the
capture state required by the selected API, but it must reproduce
leftmost-first and greedy/lazy results exactly.

### 4. Reuse match work in materializing APIs

Replacement and split currently match during both sizing and emission. Full
staged rows lost badly in the measured experiment. Any retry should store
bounded compact spans during sizing, include an overflow/rematch path, and
select it only when sampled match density and verifier cost repay scratch
traffic. The direct paths may remain faster when they rematch.

### 5. Reduce deterministic-loop instruction count

Inline byte-class ranges are implemented. Remaining comparisons include:

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
