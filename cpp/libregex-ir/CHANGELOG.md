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
- Remove the host C++ generator; retain the CPU interpreter only as test
  infrastructure.
- Add NVVM IR output, symbol prefixes, and configurable execute functions to the
  interactive explorer.
- Verify generated NVVM IR through libNVVM and nvJitLink against an nvcc LTO IR
  kernel.
- Add an optional NVBench-based cuDF-versus-NVVM GPU benchmark with warm/cold
  latency and throughput reporting.
- Add log, email, URI, and IPv4 pattern axes and publish their NVBench mean
  cold/warm results from both attached RTX A6000 GPUs.
- Add required-prefix and ASCII-literal fast paths, NVVM optimizer attributes,
  compiler-selected input loads, and explicit libNVVM/nvJitLink O3 settings.
- Determinize assertion-free boolean programs into compact Unicode-class DFA
  tables, removing recursive stack traffic and making all measured warm paths
  7x–45x faster than cuDF on the benchmark workload.
- Add pre-commit, GCC/Clang test, libFuzzer, and CUDA-toolchain CI jobs.
- Port all 92 cuDF regex-related string test functions as 543 executable case
  groups and 30 compiler assertions spanning 34,185 rows, with exact CPU and
  CUDA checks for validity, spans, captures, lists, replacements, and splits.
- Add Unicode character classes, unambiguous octal escapes, extended
  newline/CRLF behavior, and capture-enumerating generated NVVM IR.
- Consolidate all unit and fuzz tests into one file each, organize cases under
  source-named GoogleTest fixtures, and execute behavioral assertions through
  both the host interpreter and NVVM GPU JIT.
- Remove the cuDF-specific syntax dialect and bug-compatible parser behavior.
- Port cuDF's contains, count, extract, replacement, and regex-split benchmark
  matrices to single CPU/RE2 and GPU/cuDF benchmark drivers.
- Precompile test and benchmark CUDA wrappers as architecture-neutral NVCC LTO
  fatbins, remove NVRTC, and derive libNVVM/nvJitLink targets from the device.
- Profile linked kernels with Nsight Compute and Nsight Systems, fix the
  one-thread enumeration launch, inline predicate helpers, and avoid touching
  inactive capture slots.
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
