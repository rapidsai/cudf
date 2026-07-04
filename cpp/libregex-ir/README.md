# Regex IR

## What this project is about

Regex IR is a small, independent C++20 compiler for turning regular expressions
into inspectable intermediate representations and NVVM IR.
It is aimed at GPU systems that want to specialize a matcher for one pattern,
link that matcher into an existing CUDA kernel, and avoid interpreting generic
regex opcodes at runtime.

The project owns regex parsing, ordered Thompson construction, typed lowering,
optimization, verification, IR printing, deterministic specialization for
boolean matching, and textual NVVM IR generation. NVVM IR uses LLVM-derived
syntax but is NVIDIA's restricted, CUDA-specific device IR; the public API
intentionally does not claim to emit generic LLVM IR. The core library has no
LLVM, CUDA, or cuDF dependency: it assembles NVVM IR with ordinary strings and
libfmt's `fmt::format`. A consumer passes that string to libNVVM, links the
resulting PTX to its kernel with nvJitLink, then launches the kernel through its
existing CUDA runtime.

Regex IR is not a general regex runtime and it does not provide a host JIT. The
host interpreter used by the flat test suite exists only as a semantic oracle
for unit, compatibility, differential, and fuzz tests. Production execution is
the NVVM path.

    regex string
      -> ordered Thompson Automata IR
      -> typed Instruction IR
      -> optimized Instruction IR
      -> deterministic boolean executor or ordered fallback
      -> NVVM IR
      -> libNVVM PTX + nvJitLink kernel LTO IR
      -> CUDA cubin

## Build and test

libfmt is required by the core library, and GoogleTest is required when tests are
enabled. C++20 is the minimum language version.

    cmake --preset default
    cmake --build --preset default
    ctest --preset default

Or without presets:

    cmake -S . -B build -G Ninja
    cmake --build build
    ctest --test-dir build --output-on-failure

The test build requires the CUDA Toolkit. NVCC compiles the stable boolean and
capture kernels to LTO-IR fatbins at build time, the toolkit's `bin2c` embeds
those fatbins in the unit-test executable, and nvJitLink combines them with the
generated matcher PTX. The same GoogleTest fixtures execute every behavioral
case with the host interpreter and, when a device is visible, the NVVM GPU JIT.
Run the complete dual-backend suite on a selected device with:

    CUDA_VISIBLE_DEVICES=0 ./build/tests/regex_ir_tests

CPU assertions still run when no device is visible; only the explicit
`Nvvm.RuntimeAvailability` check is skipped.

## Build the documentation

The documentation builder follows the cuDF Doxygen → XML → Breathe/Sphinx
pipeline. Doxygen extracts the public C++ API and Sphinx combines it with the
Markdown guides:

    python -m pip install --requirement docs/requirements.txt
    cmake -S . -B build-docs -G Ninja \
      -DREGEX_IR_BUILD_DOCS=ON \
      -DREGEX_IR_BUILD_TESTS=OFF
    cmake --build build-docs --target regex_ir_docs

The warning-clean HTML output is written to `docs/build/html`. The equivalent
direct entry point is `make -C docs html`.

## Compile and generate NVVM IR

```cpp
#include <regex_ir.hpp>

auto compiled = regex_ir::compile(
  "abc[0-9]+", regex_ir::operation::contains());

if (compiled) {
  regex_ir::nvvm_ir_codegen_options options{
    .symbol_prefix = "tenant_17",
    .execute_function = "regex_contains_17"};
  auto nvvm_ir = regex_ir::generate_nvvm_ir(*compiled.value, options);
}
```

`symbol_prefix` is applied to generated internals so independently generated
modules do not collide. `execute_function` names the device function called by
the surrounding kernel. It has this ABI:

```llvm
define zeroext i1 @regex_contains_17(i8* %data, i64 %size)
```

Input bytes use ordinary NVVM loads so the CUDA toolchain selects the cache
behavior. If deterministic lowering is unavailable, the recursive fallback
enables required-ASCII-prefix filtering and `llvm.expect` branch hints by
default; both can be disabled through the code-generation options or the
command-line switches shown by `--help`.

NVVM IR generation supports boolean `contains`/`matches` and capture-preserving
`extract`. An extract entry uses
`i1(i8* data, i64 size, i64 search_start, i64* captures)`; successful calls
write whole-match and numbered-capture begin/end byte pairs. Calling it again
after the previous match provides the primitive used by the exact CUDA count,
findall, replacement, and split tests. The host interpreter remains the direct
executor for materialized variable-width host results.

The installed command-line generator exposes the same path:

    ./build/regex-ir-codegen \
      --symbol-prefix tenant_17 \
      --execute-function regex_contains_17 \
      'abc[0-9]+' > regex.nvvm

See [docs/usage.md](docs/usage.md) for the complete libNVVM → PTX → nvJitLink
flow, kernel declaration, execution steps, and Instruction IR mapping.

## Explore the IR

Both public IR levels have deterministic diagnostic printers:

```cpp
auto automata = regex_ir::compile_automata("a(b|c)+");
if (automata) std::cout << regex_ir::to_string(*automata.value);

auto instructions = regex_ir::compile(
  "a(b|c)+", regex_ir::operation::find());
if (instructions) std::cout << regex_ir::to_string(*instructions.value);
```

The `regex-ir-explorer` command provides an interactive regex-to-IR workflow:

    ./build/regex-ir-explorer
    regex> ^(ab|cd)+$

By default it shows Automata IR, a terminal-friendly ASCII Thompson graph,
optimized Instruction IR, and NVVM IR. Views can be selected separately:

    ./build/regex-ir-explorer --automata
    ./build/regex-ir-explorer --ir
    ./build/regex-ir-explorer --nvvm
    ./build/regex-ir-explorer --automata --ir
    ./build/regex-ir-explorer --all

It also accepts patterns non-interactively:

    ./build/regex-ir-explorer --ir --operation find 'a(b|c)+'
    ./build/regex-ir-explorer --nvvm --operation contains \
      --symbol-prefix tenant_17 --execute-function regex_17 'abc[0-9]+'

Run `./build/regex-ir-explorer --help` for all view, operation, and character-mode
options.

The GoogleTest executable accepts `--print-ir` to print the pattern, Automata
IR, and optimized Instruction IR for selected tests:

    ./build/tests/regex_ir_tests \
      --print-ir --gtest_filter=Project.SyntaxPriorityAndSpans

## Install or vendor

Installed consumers use:

```cmake
find_package(regex_ir CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE regex_ir::regex_ir)
```

For direct vendoring, the complete core library is `regex_ir.hpp` plus
`regex_ir.cpp` in the repository root. Compile the latter as C++20, link libfmt,
and expose the root as an include directory.

## CPU and GPU comparison benchmarks

The CPU driver ports the same contains, count, extract, replacement, and split
workloads and compares the test interpreter with RE2:

    cmake -S . -B build -G Ninja -DREGEX_IR_BUILD_BENCHMARKS=ON
    cmake --build build --target regex_ir_cpu_benchmark
    ./build/regex-ir-cpu-benchmark --rows 32768 --width 64
    ./build/regex-ir-cpu-benchmark --full

The GPU driver uses NVBench to compare cuDF's contains, count, extract,
replacement, and regex-split APIs with Regex IR's NVVM IR → libNVVM PTX →
nvJitLink path on the same strings column and CUDA stream. NVCC compiles the
stable benchmark wrappers to optimized LTO-IR fatbins at build time and `bin2c`
embeds them; runtime compilation is only for the generated regex, not the
wrapper. CMake uses an installed NVBench package when available and otherwise
fetches the pinned revision selected by cuDF 26.08; the fetched build requires
CMake 4.0 or newer.

    cmake -S . -B build-gpu -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DREGEX_IR_BUILD_GPU_BENCHMARKS=ON \
      -DREGEX_IR_CUDF_ROOT=/path/to/cudf/prefix
    cmake --build build-gpu --target regex_ir_gpu_benchmark
    ./build-gpu/regex-ir-gpu-benchmark \
      --devices 0 \
      --axis Rows=1000000 --axis StringBytes=128 \
      --min-samples 10 --min-time 0.5 --max-noise 1 --timeout 30 \
      --json results.json

Use `--help` for the full NVBench command line and `--list` to show devices,
benchmarks, and axes. NVBench reports isolated CPU/GPU latency, rows/second,
declared global-memory bandwidth, sample count, and noise; it also reports a
batched GPU time when the measured API permits batching.

The complete cuDF-derived registrations are `regex_ir/contains`,
`regex_ir/count`, `regex_ir/extract`, `regex_ir/replace`, `regex_ir/split`, and
matching `cudf/*` cases. Their row, width, hit-rate, capture-group, pattern, and
replacement-type axes match the cuDF benchmark matrix. The earlier warm/cold
four-pattern workload remains available as `regex_ir/warm`, `cudf/warm`,
`regex_ir/cold`, and `cudf/cold`. Combinations above cuDF's signed 32-bit
strings-offset limit are skipped.

For count, extract, replacement, and split, the Regex IR kernel measures match
and capture enumeration plus operation metadata; cuDF materializes its public
column result. This exposes matcher cost but is not an allocation-equivalent
end-to-end comparison.

The `Pattern` axis contains one literal-heavy log expression plus the email,
URI, and IPv4 expressions from the MIT-licensed
[mariomka/regex-benchmark](https://github.com/mariomka/regex-benchmark):

- `log`: `error:[ ]+[0-9]+`
- `email`: `[\w\.+-]+@[\w\.-]+\.[\w\.-]+`
- `uri`: `[\w]+://[^/\s?#]+[^\s?#]+(?:\?[^\s#]*)?(?:#[^\s]*)?`
- `ipv4`: `(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])`

The source benchmark counts non-overlapping matches in one text file. This
GPU adaptation instead performs `contains` independently per fixed-width row,
with alternating matching and non-matching rows, so its timings are not
comparable to the source project's language table. Use
`--axis Pattern=email` to select one expression.

Both engines are validated before measurement. The warm paths construct the
regex program or JIT kernel once and time repeated execution. The cold Regex IR
path parses and lowers the pattern, renders NVVM IR, runs libNVVM and
nvJitLink, loads a fresh module, and performs its first launch in every sample.
The cold cuDF path creates a fresh `regex_program` and performs its first
`contains_re` call in every sample. Input construction, host/device input
copies, and reading the precompiled LTO kernel are outside the timed region.
NVBench's own warm-up and process/library startup are also excluded, so
"cold" means per-pattern setup plus first execution, not a new-process launch.

Regex IR writes into a preallocated result buffer; cuDF's public API allocates
its result column inside the timed call. This makes the warm comparison
conservative in Regex IR's favor.

A visible NVIDIA GPU is required to run it. Results are intentionally not
portable across hardware, CUDA/cuDF versions, patterns, or input distributions.

### Measured RTX A6000 results

The following NVBench means were recorded on 2026-07-04 using two 48 GB NVIDIA
RTX A6000 GPUs, driver 595.71.05, CUDA 13.2.51, cuDF 26.08.0, GCC 13.3, and a
Release build. The warm/cold workload used 262,144 rows and 128 bytes per row
(32 MiB total), a 0.3-second minimum measurement time, at least 10 isolated
samples, 1% maximum noise, and a 30-second per-state timeout.
Latency is NVBench's mean GPU time; the cold benchmarks disable its blocking
kernel so CUDA events span host-side compilation and linking too. Speedup is
the slower latency divided by the faster latency.

| GPU | Pattern | Path | Regex IR (ms) | cuDF (ms) | Faster engine | Speedup | Regex IR (Mrows/s) | cuDF (Mrows/s) |
|---:|:---|:---|---:|---:|:---|---:|---:|---:|
| 0 | log | warm | 0.194 | 1.503 | Regex IR | 7.747x | 1,351.389 | 174.441 |
| 0 | email | warm | 0.193 | 3.577 | Regex IR | 18.560x | 1,360.395 | 73.296 |
| 0 | uri | warm | 0.194 | 3.962 | Regex IR | 20.468x | 1,354.170 | 66.161 |
| 0 | ipv4 | warm | 0.198 | 8.880 | Regex IR | 44.944x | 1,326.786 | 29.521 |
| 0 | log | cold | 10.207 | 1.511 | cuDF | 6.753x | 25.683 | 173.441 |
| 0 | email | cold | 10.344 | 3.618 | cuDF | 2.859x | 25.343 | 72.456 |
| 0 | uri | cold | 10.575 | 3.989 | cuDF | 2.651x | 24.788 | 65.720 |
| 0 | ipv4 | cold | 10.754 | 8.954 | cuDF | 1.201x | 24.377 | 29.277 |
| 1 | log | warm | 0.196 | 1.505 | Regex IR | 7.691x | 1,339.798 | 174.193 |
| 1 | email | warm | 0.196 | 3.443 | Regex IR | 17.539x | 1,335.453 | 76.142 |
| 1 | uri | warm | 0.197 | 3.804 | Regex IR | 19.275x | 1,328.253 | 68.909 |
| 1 | ipv4 | warm | 0.201 | 8.278 | Regex IR | 41.199x | 1,304.684 | 31.668 |
| 1 | log | cold | 10.035 | 1.511 | cuDF | 6.640x | 26.122 | 173.443 |
| 1 | email | cold | 10.278 | 3.476 | cuDF | 2.957x | 25.506 | 75.424 |
| 1 | uri | cold | 10.505 | 3.801 | cuDF | 2.764x | 24.955 | 68.968 |
| 1 | ipv4 | cold | 10.809 | 8.381 | cuDF | 1.290x | 24.253 | 31.279 |

The two cards were selected independently with `CUDA_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=1`. Physical GPU 0 still had an unrelated allocation of
about 2.9 GiB and produced more noise in several samples; GPU 1 was
idle and its rows are the cleaner comparison. Treat every figure as a
reproducible observation for this input distribution, not a portable claim
about arbitrary patterns or GPUs.

### Complete cuDF benchmark slice

The complete 262,144-row, 128-byte-axis slice of all five cuDF-derived
registrations was also run on the idle GPU 1: 11 contains patterns, 7 count
patterns, 3 extract group counts, 14 replacement variants, and 7 split
patterns, for 84 engine/state measurements. The minimum measurement time was
0.2 seconds with the same sample, noise, and timeout settings. Contains uses
cuDF's ten-source-row corpus and width repetition; extract uses its 100 numeric
token samples; the remaining operations use deterministic normally distributed
ASCII row widths from zero through the selected maximum.

| Family | Cases | Regex IR mean (ms) | cuDF mean (ms) | Geometric winner | Speedup | Per-case wins (Regex IR–cuDF) |
|:---|---:|---:|---:|:---|---:|:---:|
| contains | 11 | 16.937 | 2.461 | Regex IR | 3.665x | 9–2 |
| count | 7 | 78.558 | 5.518 | cuDF | 10.667x | 0–7 |
| extract | 3 | 7.142 | 5.781 | Regex IR | 1.004x | 1–2 |
| replace | 14 | 113.090 | 19.504 | cuDF | 4.417x | 0–14 |
| split | 7 | 78.949 | 15.184 | cuDF | 3.763x | 1–6 |

Arithmetic mean latency is shown only as a scale indicator; geometric speedup
weights each pattern equally. Contains patterns 0 and 1 contain positional
assertions and therefore use the recursive fallback, while the other nine use
the deterministic executor.

<details>
<summary>All 42 paired GPU-1 measurements</summary>

| Family | Case | Regex IR (ms) | cuDF (ms) | Faster engine | Speedup |
|:---|:---|---:|---:|:---|---:|
| contains | Pattern=0 | 3.932 | 1.186 | cuDF | 3.314x |
| contains | Pattern=1 | 179.962 | 7.381 | cuDF | 24.381x |
| contains | Pattern=2 | 0.763 | 1.269 | Regex IR | 1.662x |
| contains | Pattern=3 | 0.326 | 2.263 | Regex IR | 6.938x |
| contains | Pattern=4 | 0.190 | 1.738 | Regex IR | 9.154x |
| contains | Pattern=5 | 0.192 | 1.205 | Regex IR | 6.266x |
| contains | Pattern=6 | 0.187 | 3.140 | Regex IR | 16.781x |
| contains | Pattern=7 | 0.189 | 1.576 | Regex IR | 8.328x |
| contains | Pattern=8 | 0.186 | 1.496 | Regex IR | 8.032x |
| contains | Pattern=9 | 0.189 | 1.369 | Regex IR | 7.246x |
| contains | Pattern=10 | 0.185 | 4.452 | Regex IR | 24.077x |
| count | Pattern=0 | 23.870 | 5.374 | cuDF | 4.442x |
| count | Pattern=1 | 2.646 | 0.774 | cuDF | 3.418x |
| count | Pattern=2 | 82.180 | 7.397 | cuDF | 11.109x |
| count | Pattern=3 | 57.181 | 8.188 | cuDF | 6.984x |
| count | Pattern=4 | 127.986 | 7.880 | cuDF | 16.243x |
| count | Pattern=5 | 119.722 | 3.852 | cuDF | 31.084x |
| count | Pattern=6 | 136.321 | 5.158 | cuDF | 26.429x |
| extract | Groups=1 | 1.295 | 1.856 | Regex IR | 1.433x |
| extract | Groups=2 | 4.122 | 4.088 | cuDF | 1.008x |
| extract | Groups=4 | 16.010 | 11.400 | cuDF | 1.404x |
| replace | Pattern=0, Type=replace | 23.886 | 14.694 | cuDF | 1.626x |
| replace | Pattern=1, Type=replace | 2.727 | 2.155 | cuDF | 1.265x |
| replace | Pattern=2, Type=replace | 82.477 | 20.845 | cuDF | 3.957x |
| replace | Pattern=3, Type=replace | 57.600 | 22.922 | cuDF | 2.513x |
| replace | Pattern=4, Type=replace | 129.019 | 22.869 | cuDF | 5.642x |
| replace | Pattern=5, Type=replace | 119.734 | 10.618 | cuDF | 11.277x |
| replace | Pattern=6, Type=replace | 136.803 | 14.322 | cuDF | 9.552x |
| replace | Pattern=0, Type=backref | 65.337 | 22.694 | cuDF | 2.879x |
| replace | Pattern=1, Type=backref | 15.988 | 11.545 | cuDF | 1.385x |
| replace | Pattern=2, Type=backref | 165.448 | 34.846 | cuDF | 4.748x |
| replace | Pattern=3, Type=backref | 117.635 | 34.103 | cuDF | 3.449x |
| replace | Pattern=4, Type=backref | 222.299 | 28.425 | cuDF | 7.820x |
| replace | Pattern=5, Type=backref | 157.625 | 18.963 | cuDF | 8.312x |
| replace | Pattern=6, Type=backref | 286.682 | 14.048 | cuDF | 20.407x |
| split | Pattern=0 | 23.929 | 15.152 | cuDF | 1.579x |
| split | Pattern=1 | 2.795 | 3.093 | Regex IR | 1.107x |
| split | Pattern=2 | 82.563 | 20.457 | cuDF | 4.036x |
| split | Pattern=3 | 57.276 | 22.179 | cuDF | 2.582x |
| split | Pattern=4 | 129.602 | 22.080 | cuDF | 5.870x |
| split | Pattern=5 | 119.809 | 10.137 | cuDF | 11.819x |
| split | Pattern=6 | 136.671 | 13.188 | cuDF | 10.364x |

</details>

### Optimization investigation

The original linker path passed `-opt=3` to libNVVM and `-lto` to nvJitLink, but
did not pass an explicit nvJitLink optimization level. `-lto` does run the LTO
optimizer; the benchmark and integration test now also pass `-O3`, making the
requested link-time optimization level unambiguous.

The missing linker flag was not the main warm-path problem. Nsight Compute
showed the old generated kernel using 86 registers per thread, reaching only
32.8% occupancy, and saturating the L1/TEX path at about 90% while taking
roughly 2.65 ms. The renderer was launching the recursive regex block graph at
every character position and repeatedly decoding the same ASCII literal.

Before deterministic lowering, the recursive NVVM renderer was improved to:

- extract a required ASCII prefix when it is safe and skip non-candidate
  start positions before entering the block graph;
- lower fused ASCII literals to direct byte comparisons with one bounds check
  and advance their cursor with integer arithmetic;
- mark small helpers with NVVM-supported `alwaysinline`, `readonly`,
  `readnone`, and `nounwind` attributes and emit `llvm.expect` for the biased
  candidate branch;
- let the CUDA compiler select the cache behavior for ordinary input loads;
- pass `-opt=3` to libNVVM plus `-lto -O3` to nvJitLink in both the benchmark
  and the integration test.

The first optimization pass reduced the log expression to about 0.79 ms, but
complex expressions remained slower than cuDF because they still traversed the
recursive Thompson dispatcher. Nsight Compute showed the old IPv4 kernel
executing 721 million instructions and sustaining 604 GB/s of measured memory
throughput for only 32 MiB of input. Recursive call frames and cursor snapshots,
not input reads, were saturating the memory pipeline.

Boolean `contains` and `matches` do not expose captures or which prioritized
path accepted. The JIT now exploits that fact: when the Instruction IR has no
position-dependent assertions, it determinizes the graph at generation time,
partitions Unicode into predicate-equivalent alphabet classes, and emits one
compact constant transition table. Each row is streamed once with no recursive
calls, backtracking stack, or block-ID dispatcher. Assertion-bearing programs
retain the recursive fallback.

The before/after IPv4 profiles used the same 262,144 × 128-byte input:

| Nsight Compute metric | Recursive Thompson | Deterministic JIT | Change |
|:---|---:|---:|---:|
| Kernel duration | 35.47 ms | 0.201 ms | 176.5x faster |
| Executed instructions | 720.9 million | 33.9 million | 21.3x fewer |
| Warp cycles per issued instruction | 203.58 | 18.93 | 10.8x lower |
| DRAM throughput utilization | 82.81% | 22.49% | stack traffic removed |
| Configured per-thread stack | 64 KiB | 1 KiB | CUDA default restored |

The normal NVBench run, without profiler replay, measures 0.19–0.20 ms for all
four patterns. On the cleaner GPU that is 7.7x faster than cuDF for log, 17.5x
for email, 19.3x for URI, and 41.2x for IPv4. Both engines still process and
validate the same 32 MiB input.

The new cuDF count/replace/split ports exposed a separate wrapper defect: the
capture-enumeration kernel launched one thread per block. A 262,144 × 128-byte
probe took 59.16 ms. Testing 128-, 256-, 512-, 768-, and 1,024-thread blocks
selected 512 threads: it reduced that probe to 3.84 ms; 768 threads lost
residency and 1,024 exceeded the linked kernel's resource limit. The wrapper
now initializes only live capture slots, and small predicate helpers are
inlined through NVVM LTO.

Full Nsight Compute replay on the final random-width count pattern 0 workload
quantified the code-generation change independently of launch geometry:

| Nsight Compute metric | Before helper/slot pass | After | Change |
|:---|---:|---:|---:|
| Kernel duration | 24.73 ms | 24.39 ms | 1.4% faster |
| Executed instructions | 924.34 million | 841.98 million | 8.9% fewer |
| Branch instructions | 97.90 million | 79.03 million | 19.3% fewer |
| L1/TEX sectors from local memory | 97.4% | 95.4% | 2.0 points lower |
| Achieved occupancy | 31.74% | 32.83% | 1.09 points higher |
| Registers per thread | 70 | 70 | unchanged |

Nsight Systems measured two final enumeration launches at 25.29 and 27.07 ms;
host-to-device setup remained outside the measured NVBench region. PTX and SASS
inspection confirms `-opt=3`, `-lto`, and `-O3` are active and the final cubin
targets the selected device at runtime. Set `REGEX_IR_BENCHMARK_DUMP_DIR` to
write the generated `.nvvm.ll`, libNVVM `.ptx`, and linked `.cubin` artifacts
for the same inspection.

The remaining capture-path ceiling is structural, not a missing optimizer
flag. Nsight reports recursive thread-local state as 95.4% of L1/TEX sectors,
only 3.9 active threads per warp on average, and register-limited 33.3%
theoretical occupancy. The user-requested prioritized recursive executor is
preserved; cuDF's compact iterative VM therefore remains substantially faster
for most count, replacement, and split cases. Boolean patterns without
position-dependent assertions take the deterministic path and remain the JIT's
clear strength.

### Why specialized code still has branches

Pattern specialization does not imply literally branch-free matching. The DFA
loop still branches for end-of-input, ASCII versus UTF-8 decoding, and final
acceptance. Those branches are uniform or strongly biased for typical rows.
The important change is that repetition and alternation are resolved into the
current DFA state instead of recursive, input-divergent control flow.

The four generated machines are small: log uses 9 states and 7 alphabet
classes, email 7/5, URI 60/7, and IPv4 50/7. A transition is one class lookup
and one state-table lookup. The table is pattern-specific; there is no runtime
regex opcode decoder.

Cold latency still includes parsing, determinization, libNVVM compilation, PTX
generation, device LTO, nvJitLink, module loading, and the first launch. cuDF
copies a compact predefined instruction program instead of invoking a device
compiler, so it remains faster for uncached one-shot use. Production users
should cache the linked cubin by pattern, options, architecture, and toolkit
version; steady-state execution is the path where JIT specialization pays off.

## Test strategy

The default test suite includes:

- the host semantic interpreter for all operation result shapes;
- 143 boolean cases: 70 project scenarios, 20 adapted from RE2, and 53 adapted
  from Rust's `regex` crate;
- all 92 regex-related cuDF string test functions, expanded into 543 executable
  case groups, 30 compiler assertions, and 34,185 rows;
- identical exact-result checks on the host interpreter and generated NVVM CUDA
  path: validity, counts, byte spans, captures, findall lists, replacements,
  and forward/reverse split fields;
- deterministic printers and verifier mutation tests;
- optimizer differential tests;
- arbitrary parser fuzzing and exhaustive small-expression fuzzing;
- an ASan/UBSan/libFuzzer target;
- representative NVVM IR generation, real libNVVM/nvJitLink integration, and
  actual CUDA kernel launches when a device is available.

The single [unit test file](tests/unit_tests.cpp) separates cases into
`Project`, `Re2`, `RustRegex`, `Cudf`, and `Nvvm` fixtures. Every one of the 92
cuDF test names has its own `TEST_F(Cudf, ...)` entry. Stable device entry
points live in the separate [execute](tests/regex_ir_execute_kernel.cu) and
[capture](tests/regex_ir_capture_kernel.cu) CUDA wrappers and are embedded as
NVCC-generated LTO-IR fatbins. License details are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Open-ended fuzzing instructions are in [tests/FUZZING.md](tests/FUZZING.md).

## Supported syntax

- literals, UTF-8 literals, escapes, concatenation, and alternation;
- dot, character ranges/classes, negated classes, digit/word/space classes;
- beginning/end and word-boundary assertions;
- capturing and non-capturing groups;
- star, plus, optional, and counted repetitions;
- greedy and lazy priority;
- Unicode/ASCII predefined classes and case-insensitive matching;
- multiline, dot-all, extended-newline, UTF-8, and byte modes;
- three-digit octal escapes without engine-specific syntax modes.

Backreferences, lookaround, recursion, atomic groups, and engine-specific
constructs return structured unsupported-feature diagnostics.

See the [usage guide](docs/usage.md), [architecture](docs/architecture.md),
[semantics](docs/semantics.md), [IR contract](docs/ir.md), and
[code-generation guide](docs/codegen-guide.md).

## License

Apache License 2.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attributed test-suite
material.
