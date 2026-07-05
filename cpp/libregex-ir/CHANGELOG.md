# Changelog

## Unreleased

- Require C++20 and adopt cuDF's clang-format configuration.
- Use libfmt for readable textual NVVM IR assembly.
- Document every public type, member, and function and fail documentation builds
  on undocumented declarations.
- Give enums explicit `std::uint8_t` values, capitalize enumerators, use bit
  fields for consecutive boolean members, and make member defaults explicit.
- Rename the project, package, targets, headers, symbols, and public namespace
  to Regex IR and `regex_ir::`.
- Add a public CUDA-oriented NVVM IR generator plus the `regex-ir-codegen`
  command.
- Generate distinct find, count, extract, replacement, and split device ABIs;
  replacement and split support exact sizing and materialization passes.
- Remove the host C++ generator; retain the CPU interpreter only as test
  infrastructure.
- Add NVVM IR output, symbol prefixes, and configurable execute functions to the
  interactive explorer.
- Let the explorer switch contains, match, find, count, extract, replace, and
  split APIs interactively, including replacement text, view controls, active
  ABI status, and regeneration of the last pattern.
- Accept API-call expressions such as `contains("[0-9]", 12834)` in the
  explorer and execute the printed optimized IR with the CPU interpreter.
- Verify generated NVVM IR through libNVVM and nvJitLink against an nvcc LTO IR
  kernel.
- Add an optional NVBench-based cuDF-versus-NVVM GPU benchmark with warm/cold
  latency and throughput reporting.
- Add separate OpenResty, Rust Leipzig, Boost/GCC, and mariomka NVBench sources;
  fetch and checksum the complete upstream corpora, preserve Boost's long and
  medium sections, validate every state against cuDF, and report cold JIT time.
- Add log, email, URI, and IPv4 pattern axes and publish their NVBench mean
  cold/warm results from both attached RTX A6000 GPUs.
- Add required-prefix and ASCII-literal fast paths, NVVM optimizer attributes,
  compiler-selected input loads, and explicit libNVVM/nvJitLink O3 settings.
- Generate prioritized Unicode-class DFA tables for capture-free operations,
  tagged transitions for capture-safe extraction, and deterministic global
  anchor handling while retaining the recursive Thompson correctness fallback
  for non-boolean assertion/capture graphs.
- Canonicalize boolean DFA sets without unobservable path priority, use a
  15-bit boolean state index, split large boolean alternations into
  short-circuit DFA functions, and place oversized transition tables in
  read-only global memory.
- Determinize boolean boundary and line assertions with context-indexed epsilon
  closures and end-boundary acceptance, removing the recursive assertion
  fallback from contains/match kernels.
- Remove recursive stack traffic from the complete benchmark matrix and
  specialize whole-match replacement captures while both paths consume cuDF
  STRING columns and allocate equivalent owning cuDF column/table results.
- Parameterize the cuDF API matrix over 14 doubling row counts from 1,024 to
  8,388,607 and three StringBytes settings, complete all 3,528 engine states,
  and beat cuDF in all 1,764 paired RTX A6000 measurements with a 6.087x
  aggregate geometric speedup.
- Use Nsight Systems and full Nsight Compute replay to identify generic-DFA
  overhead for exact single-byte global operations and dead-state scanning in
  begin-anchored boolean machines; add direct byte search and early rejection,
  reducing the profiled split path by 3.0x–5.0x per kernel.
- Add separate reproducible 16:9 NVIDIA-themed contains, count, extract,
  replace, and split throughput slides, editable SVG assets, inline README
  tables, and case-level CSV data for every row-count/width geometry; use line
  charts exclusively for the API views.
- Keep README benchmark reporting table-first with no embedded images, and add
  linked NVIDIA-style PNG/SVG charts plus case-level CSV data for all 74
  corpus-backed OpenResty, Rust Leipzig, Boost/GCC, and mariomka expressions.
  Each plotted point retains the complete upstream source bytes.
- Add pre-commit, GCC/Clang test, libFuzzer, and CUDA-toolchain CI jobs.
- Port all 92 cuDF regex-related string test functions as 543 executable case
  groups and 30 compiler assertions spanning 34,185 rows, with exact CPU and
  CUDA checks for validity, spans, captures, lists, replacements, and splits.
- Add Unicode character classes, unambiguous octal escapes, extended
  newline/CRLF behavior, and capture-enumerating generated NVVM IR.
- Add ASCII POSIX bracket classes and the Unicode mathematical-symbol
  `\p{Sm}`/`\P{Sm}` property.
- Consolidate all unit and fuzz tests into one file each, organize cases under
  source-named GoogleTest fixtures, and execute behavioral assertions through
  both the host interpreter and NVVM GPU JIT.
- Add CPython `re_tests` and sihlfall exhaustive-category cases to the shared
  CPU/NVVM GoogleTest harness.
- Remove the cuDF-specific syntax dialect and bug-compatible parser behavior.
- Port cuDF's contains, count, extract, replacement, and regex-split benchmark
  matrices to single CPU/RE2 and GPU/cuDF benchmark drivers.
- Precompile test and benchmark CUDA wrappers as architecture-neutral NVCC LTO
  fatbins, remove NVRTC, and derive libNVVM/nvJitLink targets from the device.
- Split macro-selected test wrappers into named boolean, find, count, capture,
  replace, and split CUDA fragments under `tests/fragments/`.
- Profile linked kernels with Nsight Compute and Nsight Systems, select
  occupancy-aware 256-thread launches, write captures directly to final device
  storage, inline predicate helpers, and avoid touching inactive capture slots.
- Fix escaped ASCII literals being case-folded when case-insensitive mode was
  disabled, and remove a dead executor search-state assignment.

## 0.1.0 — 2026-07-04

Initial implementation:

- C++20 installable package and exported `regex_ir::regex_ir` target;
- bounded regex parser with structured diagnostics;
- ordered tagged Thompson Automata IR;
- typed structured-control-flow Instruction IR;
- capture specialization, epsilon folding, literal fusion, and dead-block removal;
- contains, matches, count, extract, find, replace, and split modes;
- UTF-8 logical-character and byte-offset semantics with ASCII fast paths;
- deterministic IR printers, verifiers, metrics, tests, CPU test executor, fuzz smoke, benchmark, and corpus checker;
- full Lingua Franca FSE19 robustness report.
