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

