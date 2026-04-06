# Experiment Journal — cuDF CSV Parser Optimization

## Session 1 Summary (Experiments 1-17)

Previous session applied fused parsing (single-pass delimiter scan + type conversion) to all data types sequentially. This was effective on single-type benchmarks (+75%) but is micro-benchmark tunnel vision per the updated config. The real target is `csv_read_io` (multi-type).

**Current csv_read_io HOST_BUFFER: ~3.77 GiB/s (avg of 3 runs, 6% noise range)**

Key previous discoveries:
- Shared memory atomics for valid_counts: +10% on conversion kernel
- Fused parsing for all types: +65% cumulative on single-type benchmarks
- Row offset gathering kernel: 39% of total kernel time (from nsys profiling)
- Conversion kernel: 55% of total kernel time
- Block size, tighter loops, SWAR scanning, ctxtree reuse all failed (within noise)
- SWAR integer parsing regressed (breaks single-pass fusion advantage)
- Warp-cooperative bitmask writes crashed (unsafe with divergent control flow)

---

## Session 2 begins here

### Strategic Assessment

**Portfolio**: Fused parsing captured ~65% of the +75% single-type gain. Template consolidation and NA length pre-filter added ~2%. Everything else was noise or regression.

**Bottleneck for csv_read_io**: Need to profile this specific benchmark. The multi-type case may have different bottleneck distribution than single-type due to instruction cache pressure from the large fused parser switch statement.

**Trajectory**: Diminishing returns on per-type micro-optimization. Need to pivot to architecture-level optimizations.

**Priority areas for this session**:
1. Profile csv_read_io specifically to identify THE bottleneck
2. Architecture-level: reduce total memory passes, multi-stream pipelining, host-side overhead
3. Row offset kernel optimization (39% of time in single-type, likely even more in multi-type)
4. Mixed-type warp efficiency — are threads efficiently utilizing compute in the mixed case?


## Experiment 25: __noinline__ fused parsers

**Hypothesis:** Making fused parser functions __noinline__ would reduce register pressure from 80→56, allowing more warps per SM. Spilling would decrease, improving performance.

**Result:** DISCARD — catastrophic regression (5.51 → 2.17 GiB/s, -61%). The function call overhead (save/restore registers, stack frame) is called 64 times per row (once per column), dominating execution time.

**What was learned:**
1. __noinline__ is unsuitable for hot inner-loop functions called many times per thread. The save/restore overhead is enormous.
2. The compiler's choice to inline the fused parsers (80 registers + 72 bytes spill) IS the optimal tradeoff. Inlining avoids function call overhead at the cost of register pressure.
3. The only way to reduce register pressure without __noinline__ is to reduce code complexity — i.e., fewer fused parser branches. But that means losing the per-type optimization that gave the +75% single-type improvement.
4. **Conclusion: The current register/spill balance is locally optimal given the fused parsing architecture.** Further kernel improvement requires a fundamentally different approach (like column-oriented processing instead of row-oriented).

## Circuit Breaker — Deep Research Mode (after Exp22, 24, 25 failures)

Three consecutive discards:
- Exp22: skip blank row removal (0.1ms too small to detect)
- Exp24: register limit (increased spill → -34%)
- Exp25: __noinline__ (function call overhead → -61%)

**Root cause**: All three targeted symptoms of the same underlying issue — the conversion kernel is at its local optimum for the current row-oriented, fused-parsing architecture. Register pressure, spilling, and code footprint are intrinsically coupled by the large switch statement.

**What NOT to try next:**
- Any register/occupancy tuning (Exp24/25 proved this is counterproductive)
- Micro-optimizations to individual fused parsers (diminishing returns since Exp12)
- Per-kernel parameter changes (block size, shared memory, etc.)

**What MIGHT work:**
- Column-oriented processing (ParPaRaw's approach): assign threads to columns instead of rows, so each thread only has one type's code path → minimal register pressure, no warp divergence
- Host-side optimization: reduce the 11ms of non-GPU overhead
- Multi-stream pipelining: overlap H2D with compute
- Writer optimization: completely untouched

Spawning research agents for fresh ideas before the next experiment.

## Experiment 26: Column-oriented split conversion (cancelled)

**Hypothesis:** Split the monolithic conversion kernel into per-type sub-kernels to reduce register pressure from 80 to ~30 per sub-kernel.

**Analysis:** Even with `if constexpr` gating the fused parser branches, the `type_dispatcher` fallback path (which handles all types) is compiled in every instantiation. This means every sub-kernel still has ALL type paths in its binary, keeping register pressure high. True column-oriented conversion requires replacing the entire kernel architecture (field index + per-type kernels without type_dispatcher fallback), which is a multi-day engineering effort.

**Decision:** Deferred. The column-oriented approach is the theoretically correct solution but requires too much refactoring for one experiment cycle. Need to find a smaller incremental step.

## Note: Benchmark variability across sessions

The csv_read_io HOST_BUFFER result dropped from 5.51 GiB/s (earlier in this session) to 3.92 GiB/s (current). The code hasn't changed (same commit d4d3b515). GPU SM clock verified at 2405 MHz (near max 3003 MHz). The likely cause is system-level variability (thermal, power management, background processes). 

**Going forward:** Use RELATIVE improvements between consecutive runs (same session conditions) rather than absolute numbers for experiment decisions. Run back-to-back comparisons on the same session for reliable results.

## Experiment 31: L2 persistent cache for NA trie

**Hypothesis:** Pin the NA trie in L2 persistent cache using cudaAccessPolicyWindow to prevent eviction by streaming CSV data.

**Result:** DISCARD — within noise (5.33 vs 5.29 GiB/s). The NA trie is ~400 bytes, well within L1 cache (128KB/SM). L1 already keeps it hot without L2 pinning.

**What was learned:** L2 cache pinning only helps for data that's too large for L1 but accessed repeatedly. At ~400 bytes, the NA trie lives comfortably in L1. The L2 persistent API adds host-side overhead (cudaStreamSetAttribute calls) that may actually slow things down for small structures.

## Strategic Conclusion After 31 Experiments

The CSV reader optimization has reached a definitive plateau at ~5.3 GiB/s (+145% from baseline). The last 12 experiments (Exp22-31) produced ZERO measurable improvements despite trying:
- Register pressure reduction (Exp24, 25, 27, 28, 30) 
- Host overhead reduction (Exp22, 29)
- Cache optimization (Exp31)
- Various novel approaches from research

The bottleneck is ARCHITECTURAL: the row-oriented, monolithic-kernel design with 80-register pressure. The only path to significantly better performance is column-oriented conversion (CUDAFastCSV approach), which requires a major refactor.

**Recommendations for future work:**
1. Column-oriented conversion with field index (CUDAFastCSV approach) — multi-day effort, estimated 2-3x potential gain
2. Writer optimization — completely untouched, 2.5x slower than reader
3. Multi-stream pipelining for FILEPATH path — moderate effort, 10-20% potential on file I/O
