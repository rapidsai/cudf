# Regex IR

## What this project is about

Regex IR is a small, independent C++20 compiler for turning regular expressions
into inspectable intermediate representations and NVVM IR.
It is aimed at GPU systems that want to specialize a matcher for one pattern,
link that matcher into an existing CUDA kernel, and avoid interpreting generic
regex opcodes at runtime.

The project owns regex parsing, ordered Thompson construction, typed lowering,
optimization, verification, IR printing, prioritized and tagged deterministic
specialization, a gated bit-parallel Glushkov executor, and textual NVVM IR
generation. NVVM IR uses LLVM-derived syntax but is NVIDIA's restricted,
CUDA-specific device IR; the public API intentionally does not claim to emit
generic LLVM IR. The core library has no LLVM, CUDA, or cuDF dependency: it
assembles NVVM IR with ordinary strings and libfmt's `fmt::format`. A consumer
passes that string to libNVVM with `-gen-lto`, links the resulting LTO-IR to its
kernel LTO-IR with nvJitLink, then launches the cubin through its existing CUDA
runtime.

Regex IR is not a general regex runtime and it does not provide a host JIT. The
host interpreter used by the flat test suite exists only as a semantic oracle
for unit, compatibility, differential, and fuzz tests. Production execution is
the NVVM path.

See [optimization.md](optimization.md) for the current executor-selection
rules, IR and NVVM optimizations, profiling conclusions, and remaining
performance opportunities.

    regex string
      -> ordered Thompson Automata IR
      -> typed Instruction IR
      -> optimized Instruction IR
      -> Glushkov NFA, prioritized/tagged DFA, or ordered Thompson fallback
      -> NVVM IR
      -> libNVVM LTO IR + NVCC kernel LTO IR
      -> nvJitLink whole-program device LTO
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

The test build requires the CUDA Toolkit. NVCC compiles stable boolean, find,
count, capture, replace, and split adapters to LTO-IR fatbins at build time,
the toolkit's `bin2c` embeds those fatbins in the unit-test executable, and
nvJitLink combines them with generated operation-specific libNVVM LTO IR. The same
GoogleTest fixtures execute every behavioral case with the host interpreter
and, when a device is visible, the NVVM GPU JIT.
Run the complete dual-backend suite on a selected device with:

    CUDA_VISIBLE_DEVICES=0 ./build/tests/regex_ir_tests

CPU assertions still run when no device is visible; only the explicit NVVM
runtime-launch checks are skipped.

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
behavior. Selected assertion-free boolean programs use one 64-bit Glushkov
position state when it avoids a large transition table or represents a long
linear follow graph; other capture-free programs use deterministic states when
the state-table resource limits permit it. Capture programs use tagged
deterministic transitions when their capture history is provably unambiguous;
the recursive fallback enables required-ASCII-prefix filtering and
`llvm.expect` branch hints. Both fallback hints can be disabled through the
code-generation options or the command-line switches shown by `--help`.

NVVM generation specializes the public entry point for the selected API:

| Operation | Generated device ABI |
|:---|:---|
| `contains`, `matches` | `i1(i8* data, i64 size)` |
| `find` | `i1(i8* data, i64 size, i64* span)` |
| `count` | `i64(i8* data, i64 size)` |
| `extract` | `i1(i8* data, i64 size, i64 search_start, i64* captures)` |
| `replace` | `i64(i8* data, i64 size, i8* output)` |
| `split` | `i64(i8* data, i64 size, i64* spans)` |

Replace returns its exact output byte count and split returns its exact field
count. Passing a null output performs a sizing pass; passing storage emits the
replacement bytes or split begin/end pairs. Count, replacement expansion, and
split iteration are generated into their own NVVM functions and no longer run
through the extract adapter. The host interpreter remains the semantic oracle.

The installed command-line generator exposes the same path:

    ./build/regex-ir-codegen \
      --symbol-prefix tenant_17 \
      --execute-function regex_contains_17 \
      'abc[0-9]+' > regex.nvvm

See [docs/usage.md](docs/usage.md) for the complete libNVVM LTO-IR → nvJitLink
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
    regex-ir[contains]> ^(ab|cd)+$

Use function-call input to compile a regex and immediately execute the printed
Instruction IR with the CPU interpreter:

    regex-ir[contains]> contains("[0-9]", 12834)
    input = "12834"
    contains = true

The first argument is a quoted regex. The second argument is the input text;
an unquoted scalar such as `12834` is treated as the text `"12834"`. Quoted
input supports spaces, commas, and escapes. All APIs use the same form:

    match("[0-9]+", 12834)
    find("[a-z]+", "123abc")
    count("[0-9]", 12834)
    extract("([a-z]+)([0-9]+)", "ab12")
    replace("([0-9]+)", "id=42", "<$1>")
    split("[,;]+", "a,b;c")

The result section prints booleans, byte spans and matched text, counts,
capture spans, replacement text, or split fields as appropriate. A bare regex
continues to compile and print without invoking the interpreter. Function-call
input also works non-interactively when the whole call is shell-quoted:

    ./build/regex-ir-explorer --ir 'contains("[0-9]", 12834)'

By default it shows Automata IR, a terminal-friendly ASCII Thompson graph,
optimized Instruction IR, and NVVM IR. Views can be selected separately:

    ./build/regex-ir-explorer --automata
    ./build/regex-ir-explorer --ir
    ./build/regex-ir-explorer --nvvm
    ./build/regex-ir-explorer --automata --ir
    ./build/regex-ir-explorer --all

It also accepts patterns non-interactively:

    ./build/regex-ir-explorer --ir --operation find 'a(b|c)+'
    ./build/regex-ir-explorer --nvvm --operation match '^(ab|cd)+$'
    ./build/regex-ir-explorer --nvvm --operation count '[0-9]+'
    ./build/regex-ir-explorer --nvvm --operation extract '([a-z]+)([0-9]+)'
    ./build/regex-ir-explorer --nvvm --operation replace \
      --replacement '<$1>' '([a-z]+)'
    ./build/regex-ir-explorer --nvvm --operation split '[,;]+'
    ./build/regex-ir-explorer --nvvm --operation contains \
      --symbol-prefix tenant_17 --execute-function regex_17 'abc[0-9]+'

The API can also be changed without leaving the console. The last pattern is
regenerated immediately with the new result shape and device ABI:

    regex-ir[contains]> :operation count
    regex-ir[count]> :replace <$1>
    regex-ir[replace]> :show nvvm
    regex-ir[replace]> :hide automata
    regex-ir[replace]> :status

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
replacement, and regex-split APIs with Regex IR's NVVM IR → libNVVM LTO-IR →
nvJitLink path on the same strings column and CUDA stream. NVCC compiles the
stable benchmark wrappers to optimized LTO-IR fatbins at build time and `bin2c`
embeds them; runtime compilation is only for the generated regex, not the
wrapper. CMake uses an installed NVBench package when available and otherwise
fetches the pinned revision selected by cuDF 26.08; the fetched build requires
CMake 4.0 or newer.

The benchmark registrations are intentionally separate because they answer
different questions:

| Suite | Registrations | Setup | Intended measurement |
|:---|:---|:---|:---|
| cuDF API matrix | `regex_ir/{contains,count,extract,replace,split}` and `cudf/*` | One strings column; cuDF-derived row, width, hit-rate, capture, and replacement axes | End-to-end behavior of each public regex API, including materialization for transforms |
| warm/cold JIT | `regex_ir/{warm,cold}` and `cudf/{warm,cold}` | Four log/email/URI/IPv4 expressions over fixed-width rows | Cached execution, uncached JIT readiness, and setup plus first launch |
| OpenResty | `regex_ir/openresty`, `cudf/openresty` | 31 expressions over the full-size `abc`, random alphabet, delimiter, and `mtent12` inputs | Literal misses, alternation, classes, assertions, captures, and bounded/greedy/lazy repetition |
| Rust Leipzig | `regex_ir/leipzig`, `cudf/leipzig` | All 18 expressions over the complete 16,013,977-byte `3200.txt` | Real English scanning, bounded repetition, Unicode literals/properties, and the CSV stress expression |
| Boost/GCC | `regex_ir/boost`, `cudf/boost` | 36 long/medium Twain, complete `crc.hpp`, complete `libraries.htm`, and exact scalar cases | Domain diversity and the difference between long scans, short files, and exact row-shaped matching |
| mariomka | `regex_ir/mariomka`, `cudf/mariomka` | All three expressions over the complete 6,839,410-byte `input-text.txt` | Email, URI, and IPv4 scanning in mixed prose and source code |

    cmake -S . -B build-gpu -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DREGEX_IR_BUILD_GPU_BENCHMARKS=ON \
      -DREGEX_IR_CUDF_ROOT=/path/to/cudf/prefix
    cmake --build build-gpu \
      --target regex_ir_gpu_benchmark regex_ir_corpus_benchmark
    ./build-gpu/regex-ir-gpu-benchmark \
      --devices 0 --min-samples 5 --min-time 0.05 --max-noise 2 --timeout 30 \
      --benchmark regex_ir/contains \
        --axis 'StringBytes=[64,128,256]' --axis HitRate=50 \
      --benchmark cudf/contains \
        --axis 'StringBytes=[64,128,256]' --axis HitRate=50 \
      --json cudf-contains.json
    ./build-gpu/regex-ir-gpu-benchmark \
      --devices 0 --min-samples 5 --min-time 0.05 --max-noise 2 --timeout 30 \
      --benchmark regex_ir/count \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark cudf/count \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark regex_ir/extract \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark cudf/extract \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark regex_ir/replace \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark cudf/replace \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark regex_ir/split \
        --axis 'StringBytes=[64,128,256]' \
      --benchmark cudf/split \
        --axis 'StringBytes=[64,128,256]' \
      --json cudf-transforms.json
    MPLCONFIGDIR=/tmp/regex-ir-matplotlib \
      python3 scripts/plot_cudf_benchmarks.py \
        cudf-contains.json cudf-transforms.json \
        --readme README.md

The four imported corpus suites live in separate source files and run from a
second executable. Building the target downloads missing corpora into the
build tree and verifies every file before use:

    ./build-gpu/regex-ir-corpus-benchmark \
      --devices 0 --min-samples 5 --min-time 0.05 --max-noise 2 --timeout 10 \
      --benchmark regex_ir/openresty \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark cudf/openresty \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark regex_ir/leipzig \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark cudf/leipzig \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark regex_ir/boost --axis 'Case=[1,2,3,4,5,6]' \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark cudf/boost --axis 'Case=[1,2,3,4,5,6]' \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark regex_ir/mariomka \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --benchmark cudf/mariomka \
        --axis Rows=32768 --axis Columns=8 --axis MaxStringBytes=256 \
      --json corpus-large.json
    ./build-gpu/regex-ir-corpus-benchmark \
      --devices 0 --min-samples 5 --min-time 0.05 --max-noise 2 --timeout 10 \
      --benchmark regex_ir/boost \
        --axis 'Case=[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]' \
        --axis Rows=1024 --axis Columns=1 --axis MaxStringBytes=64 \
      --benchmark cudf/boost \
        --axis 'Case=[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]' \
        --axis Rows=1024 --axis Columns=1 --axis MaxStringBytes=64 \
      --json corpus-compact.json
    MPLCONFIGDIR=/tmp/regex-ir-matplotlib \
      python3 scripts/plot_corpus_benchmarks.py \
        corpus-large.json corpus-compact.json --readme README.md

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

Count invokes a generated prioritized-DFA count function, and capture-safe
extract expressions use tagged DFA transitions once per row. Replacement and
split time their complete sizing, device prefix-scan, and emission passes,
then construct owning cuDF STRING or LIST<STRING> columns. Contains and count
likewise allocate owning BOOL8 and INT32 columns. Both implementations receive
the same owning cuDF STRING input column, and both time output allocation and
materialization.

The `Pattern` axis contains one literal-heavy log expression plus the email,
URI, and IPv4 expressions from the MIT-licensed
[mariomka/regex-benchmark](https://github.com/mariomka/regex-benchmark):

- `log`: `error:[ ]+[0-9]+`
- `email`: `[\w\.+-]+@[\w\.-]+\.[\w\.-]+`
- `uri`: `[\w]+://[^/\s?#]+[^\s?#]+(?:\?[^\s#]*)?(?:#[^\s]*)?`
- `ipv4`: `(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])`

The source benchmark counts non-overlapping matches in one text file. The
fixed-row `Pattern` workload still uses generated matching/non-matching rows,
so its timings are not comparable to the source language table. The separate
`regex_ir/mariomka` and `cudf/mariomka` registrations below use the complete
upstream file. Use `--axis Pattern=email` to select the synthetic API workload,
or `--benchmark regex_ir/mariomka --axis Case=1` for the source corpus.

### Imported corpus construction

The external driver preserves the expression inventories and workload roles
from the [31-case OpenResty comparison](https://openresty.org/misc/re/bench/),
the Apache-2.0
[rust-leipzig/regex-performance](https://github.com/rust-leipzig/regex-performance)
suite, Boost.Regex's
[GCC performance comparison](https://www.boost.org/doc/libs/1_41_0/libs/regex/doc/gcc-performance.html),
and [mariomka/regex-benchmark](https://github.com/mariomka/regex-benchmark).
It does not claim that row-wise GPU timings are comparable to those historical
CPU results: the source suites scan monolithic files, whereas this adaptation
measures `contains` independently for every strings-column row.

For file-backed cases, the driver partitions every source byte exactly once
across the requested table. Boundaries prefer nearby newlines, never split a
UTF-8 code point, vary deterministically around the remaining average, and do
not inject matches. A state is skipped when `Rows * Columns * MaxStringBytes`
cannot hold the complete input. The scalar Boost cases repeat their complete
source string once per row, matching the original benchmark's repeated-match
measurement.

OpenResty's `abc.txt`, `rand-abc.txt`, and `delim.txt` are generated in memory
at their original 25 MiB/10 MiB sizes. Its historical Perl scripts did not set
a random seed, so the latter two retain the exact size and construction recipe
with a fixed generator. All Twain cases use the exact 20,045,118-byte file from
the mirror linked by OpenResty. Boost's medium section separately uses the
first 50,000 bytes, while its long section uses the whole file; no duplicate
section is collapsed.

The complete corpus manifest is:

| Dataset | Bytes | SHA-256 |
|:---|---:|:---|
| generated OpenResty `abc.txt` | 26,214,408 | `e88eb171d3e31edf2f4545dc5158a5303207c7765968a9d6732451271ebe934e` |
| generated OpenResty `rand-abc.txt` | 10,485,768 | `d246c82e81eb6a3633e3a00ba4348fa9c45e357e03c7520e0aaaa08d6574015b` |
| generated OpenResty `delim.txt` | 10,485,769 | `8f30e65d34dad8cc9447981d191d72fd04ca215dc92ffdb5791fb19355f26166` |
| OpenResty/Boost `mtent12.txt` | 20,045,118 | `0bdd71ad57eb2224a21ea39f19e636e4208ee3cd3d0d77cf1fe8b22ed58ed5eb` |
| Rust Leipzig `3200.txt` | 16,013,977 | `f2aa28234e7a8212c9e009fa9c67d1960d2d063d076765de46b0faed5fe44ad8` |
| mariomka `input-text.txt` | 6,839,410 | `7b7f70c9ca999b2bede85b7ed8e37c9193edced196f4aed29651e37ef4f8e979` |
| Boost 1.41 `boost/crc.hpp` | 34,483 | `21a321a85fa867bb6b8f2d37f9159be4ea0807d77627e83e89ca1f36c487c954` |
| Boost 1.41 `libs/libraries.htm` | 51,799 | `4446858edb8ce420f0372b9043c1955be79747efbeceee415faae01491dc06cf` |

Set `REGEX_IR_BENCHMARK_CORPUS_DIRECTORY` at configure time to reuse an
offline corpus cache. The build still verifies all five hashes.

Both engines consume identical columns, use the same CUDA stream, and must
produce identical boolean output before sampling begins. Pattern compilation,
libNVVM, nvJitLink, corpus construction, host/device copies, and validation are
outside the execution-latency sample. Every Regex IR state separately reports
uncached `JIT Ready` time from the source regex through parsing/lowering, NVVM
IR rendering, libNVVM LTO-IR generation, nvJitLink, module loading, and kernel
function lookup. Corpus construction and the first launch are excluded. cuDF
states report `regex_program::create` separately. Both engines consume the
same owning cuDF STRING columns and allocate owning cuDF result objects inside
the timed call.

Cross-engine agreement is the automatic pre-timing guard, not the sole
semantic proof for an outlier. The
[independent-oracle protocol](docs/usage.md#oracle-correctness-protocol-for-benchmarks)
freezes the source expression and row partition, compares the complete result
shape against a separate semantic engine, and records a reproducible vector
hash when validation spans processes. It includes the audit of Boost/GCC case
14 (`cpp tokens`).

Expressions are retained when the target syntax supports them. Boost's
`\<`/`\>` boundary extension is normalized to standard `\b` for both engines.
Regex IR accepts the source POSIX classes; cuDF receives equivalent ASCII
classes, and the large Boost tokenizer has a portable cuDF spelling for its
nested POSIX-class union. Leipzig's exact `\p{Sm}` is compiled by Regex IR,
while cuDF receives `[+<=>|~∞]`, the mathematical-symbol subset present in the
source dataset, because cuDF rejects Unicode property escapes. Source
`(?i)` prefixes are represented by each engine's case-insensitive compile flag.

Both engines are validated before measurement. The warm paths construct the
regex program or JIT kernel once and time repeated execution. The cold Regex IR
path parses and lowers the pattern, renders NVVM IR, runs libNVVM and
nvJitLink, loads a fresh module, and performs its first launch in every sample.
The cold cuDF path creates a fresh `regex_program` and performs its first
`contains_re` call in every sample. Input construction, host/device input
copies, and reading the precompiled LTO kernel are outside the timed region.
NVBench's own warm-up and process/library startup are also excluded, so
"cold" means per-pattern setup plus first execution, not a new-process launch.

The result contract is identical for both paths:

| API | Timed owning result |
|:---|:---|
| contains | cuDF `BOOL8` column |
| count | cuDF `INT32` column |
| extract | cuDF table of `STRING` columns |
| replace | cuDF `STRING` column |
| split | cuDF `LIST<STRING>` column |

Regex IR's wrappers write directly into cuDF-owned fixed-width or strings
children. Extract and split use cuDF's strings factories to compact generated
pointer/length pairs. Every setup result is compared recursively with cuDF,
including nested child offsets and bytes, before timing begins.

A visible NVIDIA GPU is required to run it. Results are intentionally not
portable across hardware, CUDA/cuDF versions, patterns, or input distributions.

### Suite A results: cuDF API matrix

The presentation set is generated directly from the final NVBench JSON and is
split by regex API. Each API below has one line chart, its full parameterized
results table, an editable SVG, and case-level CSV data.
The line-chart panels select `StringBytes`; their x-axis is row count and their
two lines compare Regex IR with cuDF.

The complete Regex IR matrix was rerun on 2026-07-07 with the current code.
Because cuDF was unchanged, its comparison states come from the immediately
preceding full run on the same GPU and software stack; every candidate Regex IR
state was replayed and validated against cuDF before timing.
It uses 14 doubling row counts from 1,024 through 8,388,607 and `StringBytes`
settings 64, 128, and 256: 42 input geometries for every expression. The upper
bound is the largest common row count that keeps a 256-byte input within cuDF's
signed 32-bit string-offset limit. The expression matrix is 11 contains,
7 count, 1/2/4-group extract, 7 plain plus 7 backreference replacements, and
7 split cases. In total, the run completed 1,764 paired comparisons and 3,528
engine/geometry states. Contains uses its 50% hit-rate axis.

`StringBytes` is the benchmark's input-width parameter rather than a claim that
every row has exactly that length. Contains repeats the cuDF source-row corpus
in proportion to it; count, replace, and split sample variable lengths capped
at it; extract generates rows exactly that wide. The README tables retain the
axis name so this distinction stays visible.

The system used an otherwise idle 48 GB NVIDIA RTX A6000, driver 595.71.05,
CUDA 13.2.51, cuDF 26.08.0, GCC 13.3, and a Release build. Every state used at
least five samples, 0.05 seconds minimum measurement time, a 2% maximum-noise
target, and a 30-second timeout.

These are steady-state execution measurements: pattern compilation, libNVVM,
nvJitLink, module loading, and input-column construction are outside the timed
region. Output allocation and complete cuDF-column materialization are inside
both timed paths. Replacement includes Regex IR's sizing kernel, device prefix
scan, exact chars allocation, and emission. Split additionally builds the list
offsets and compact STRING child; extract builds the full table of capture
STRING columns. Geometric throughput and speedup weight every
expression/geometry pair equally.

<!-- BEGIN GENERATED CUDF API RESULTS -->

| API | Expressions | Geometries | Paired measurements | Regex IR geometric throughput (M rows/s) | cuDF geometric throughput (M rows/s) | Geometric speedup | Pair wins | Regex IR JIT-ready mean (ms) |
|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| contains | 11 | 42 | 462 | 526.870 | 66.241 | 7.954x | 462–0 | 23.387 |
| count | 7 | 42 | 294 | 223.884 | 24.817 | 9.021x | 294–0 | 28.734 |
| extract | 3 | 42 | 126 | 94.160 | 32.747 | 2.875x | 126–0 | 43.277 |
| replace | 14 | 42 | 588 | 71.328 | 7.086 | 10.066x | 588–0 | 62.984 |
| split | 7 | 42 | 294 | 41.944 | 9.341 | 4.490x | 294–0 | 59.428 |
| **All APIs** | **42** | **42** | **1764** | **136.046** | **18.317** | **7.427x** | **1764–0** | **44.905** |

Regex IR won 1764 of 1764 paired measurements. Contains, count, extract, replace, and
split won every state. The narrowest result was contains pattern 3 at 131,072 rows/256
bytes (1.056x), while the largest was contains pattern 10 at 8,388,607 rows/256 bytes
(35.975x). The results describe this hardware, corpus, and allocation contract—not every
regex or deployment. Use the commands above to reproduce the matrix.

The tables below report geometric mean throughput across each API's expression
cases independently at every row-count/`StringBytes` geometry. JIT-ready is the
arithmetic mean from regex string through module load and kernel-function lookup;
it excludes input construction and the first launch.

#### Contains

[PNG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-contains-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins | Regex IR JIT-ready mean (ms) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,024 | 64 | 11 | 37.874 | 8.388 | 4.515x | 11–0 | 27.787 |
| 1,024 | 128 | 11 | 27.315 | 5.338 | 5.118x | 11–0 | 22.940 |
| 1,024 | 256 | 11 | 19.209 | 3.370 | 5.701x | 11–0 | 23.125 |
| 2,048 | 64 | 11 | 76.075 | 14.056 | 5.412x | 11–0 | 22.538 |
| 2,048 | 128 | 11 | 54.294 | 9.067 | 5.988x | 11–0 | 22.497 |
| 2,048 | 256 | 11 | 38.209 | 6.119 | 6.245x | 11–0 | 22.576 |
| 4,096 | 64 | 11 | 151.144 | 29.478 | 5.127x | 11–0 | 22.610 |
| 4,096 | 128 | 11 | 107.362 | 19.966 | 5.377x | 11–0 | 22.730 |
| 4,096 | 256 | 11 | 75.870 | 13.071 | 5.804x | 11–0 | 22.995 |
| 8,192 | 64 | 11 | 297.210 | 48.685 | 6.105x | 11–0 | 22.581 |
| 8,192 | 128 | 11 | 207.582 | 27.821 | 7.461x | 11–0 | 22.845 |
| 8,192 | 256 | 11 | 148.330 | 18.875 | 7.859x | 11–0 | 22.739 |
| 16,384 | 64 | 11 | 575.413 | 66.694 | 8.628x | 11–0 | 22.280 |
| 16,384 | 128 | 11 | 398.669 | 48.220 | 8.268x | 11–0 | 22.499 |
| 16,384 | 256 | 11 | 285.623 | 36.429 | 7.840x | 11–0 | 23.215 |
| 32,768 | 64 | 11 | 932.014 | 91.614 | 10.173x | 11–0 | 22.550 |
| 32,768 | 128 | 11 | 640.123 | 70.286 | 9.107x | 11–0 | 22.956 |
| 32,768 | 256 | 11 | 454.406 | 47.543 | 9.558x | 11–0 | 25.490 |
| 65,536 | 64 | 11 | 1,580.342 | 114.912 | 13.753x | 11–0 | 22.881 |
| 65,536 | 128 | 11 | 1,073.828 | 87.868 | 12.221x | 11–0 | 22.955 |
| 65,536 | 256 | 11 | 729.495 | 67.379 | 10.827x | 11–0 | 22.598 |
| 131,072 | 64 | 11 | 1,923.757 | 132.459 | 14.523x | 11–0 | 26.187 |
| 131,072 | 128 | 11 | 712.034 | 97.046 | 7.337x | 11–0 | 22.549 |
| 131,072 | 256 | 11 | 376.653 | 73.779 | 5.105x | 11–0 | 22.985 |
| 262,144 | 64 | 11 | 2,516.678 | 159.843 | 15.745x | 11–0 | 24.025 |
| 262,144 | 128 | 11 | 1,046.445 | 114.034 | 9.177x | 11–0 | 22.991 |
| 262,144 | 256 | 11 | 553.152 | 82.815 | 6.679x | 11–0 | 23.246 |
| 524,288 | 64 | 11 | 2,971.406 | 253.854 | 11.705x | 11–0 | 22.882 |
| 524,288 | 128 | 11 | 1,384.080 | 169.300 | 8.175x | 11–0 | 23.781 |
| 524,288 | 256 | 11 | 743.417 | 125.404 | 5.928x | 11–0 | 24.381 |
| 1,048,576 | 64 | 11 | 3,186.652 | 252.712 | 12.610x | 11–0 | 24.695 |
| 1,048,576 | 128 | 11 | 1,681.616 | 179.776 | 9.354x | 11–0 | 23.450 |
| 1,048,576 | 256 | 11 | 921.907 | 126.638 | 7.280x | 11–0 | 24.065 |
| 2,097,152 | 64 | 11 | 3,008.387 | 277.505 | 10.841x | 11–0 | 22.925 |
| 2,097,152 | 128 | 11 | 1,751.644 | 190.122 | 9.213x | 11–0 | 22.944 |
| 2,097,152 | 256 | 11 | 1,017.252 | 138.727 | 7.333x | 11–0 | 23.741 |
| 4,194,304 | 64 | 11 | 3,085.694 | 370.373 | 8.331x | 11–0 | 23.159 |
| 4,194,304 | 128 | 11 | 1,858.304 | 247.866 | 7.497x | 11–0 | 22.745 |
| 4,194,304 | 256 | 11 | 1,100.921 | 137.790 | 7.990x | 11–0 | 23.080 |
| 8,388,607 | 64 | 11 | 3,119.089 | 312.627 | 9.977x | 11–0 | 22.562 |
| 8,388,607 | 128 | 11 | 1,913.936 | 240.144 | 7.970x | 11–0 | 22.782 |
| 8,388,607 | 256 | 11 | 1,149.864 | 176.341 | 6.521x | 11–0 | 27.673 |

#### Count

[PNG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-count-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins | Regex IR JIT-ready mean (ms) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,024 | 64 | 7 | 18.641 | 2.632 | 7.083x | 7–0 | 28.992 |
| 1,024 | 128 | 7 | 10.591 | 1.408 | 7.521x | 7–0 | 28.782 |
| 1,024 | 256 | 7 | 6.145 | 0.730 | 8.421x | 7–0 | 28.541 |
| 2,048 | 64 | 7 | 35.665 | 5.147 | 6.929x | 7–0 | 28.498 |
| 2,048 | 128 | 7 | 20.673 | 2.767 | 7.473x | 7–0 | 28.783 |
| 2,048 | 256 | 7 | 11.549 | 1.382 | 8.356x | 7–0 | 29.185 |
| 4,096 | 64 | 7 | 70.814 | 10.091 | 7.018x | 7–0 | 28.549 |
| 4,096 | 128 | 7 | 40.034 | 5.414 | 7.395x | 7–0 | 28.110 |
| 4,096 | 256 | 7 | 22.464 | 2.770 | 8.111x | 7–0 | 28.338 |
| 8,192 | 64 | 7 | 135.015 | 19.716 | 6.848x | 7–0 | 28.987 |
| 8,192 | 128 | 7 | 78.073 | 10.682 | 7.309x | 7–0 | 28.693 |
| 8,192 | 256 | 7 | 44.562 | 5.488 | 8.120x | 7–0 | 29.329 |
| 16,384 | 64 | 7 | 267.402 | 37.080 | 7.211x | 7–0 | 28.669 |
| 16,384 | 128 | 7 | 149.150 | 20.091 | 7.424x | 7–0 | 28.626 |
| 16,384 | 256 | 7 | 86.259 | 10.849 | 7.951x | 7–0 | 29.363 |
| 32,768 | 64 | 7 | 472.435 | 58.295 | 8.104x | 7–0 | 28.621 |
| 32,768 | 128 | 7 | 283.303 | 34.575 | 8.194x | 7–0 | 28.244 |
| 32,768 | 256 | 7 | 156.576 | 18.014 | 8.692x | 7–0 | 28.236 |
| 65,536 | 64 | 7 | 729.841 | 86.834 | 8.405x | 7–0 | 27.962 |
| 65,536 | 128 | 7 | 439.338 | 48.549 | 9.049x | 7–0 | 28.137 |
| 65,536 | 256 | 7 | 235.979 | 25.310 | 9.324x | 7–0 | 30.164 |
| 131,072 | 64 | 7 | 803.235 | 100.306 | 8.008x | 7–0 | 28.318 |
| 131,072 | 128 | 7 | 485.609 | 56.350 | 8.618x | 7–0 | 28.690 |
| 131,072 | 256 | 7 | 195.855 | 27.739 | 7.061x | 7–0 | 28.303 |
| 262,144 | 64 | 7 | 1,163.340 | 102.252 | 11.377x | 7–0 | 28.668 |
| 262,144 | 128 | 7 | 673.106 | 58.838 | 11.440x | 7–0 | 29.603 |
| 262,144 | 256 | 7 | 266.875 | 29.862 | 8.937x | 7–0 | 28.920 |
| 524,288 | 64 | 7 | 1,296.778 | 115.864 | 11.192x | 7–0 | 32.335 |
| 524,288 | 128 | 7 | 754.906 | 64.608 | 11.684x | 7–0 | 28.180 |
| 524,288 | 256 | 7 | 320.335 | 32.676 | 9.803x | 7–0 | 27.972 |
| 1,048,576 | 64 | 7 | 1,412.302 | 123.942 | 11.395x | 7–0 | 27.885 |
| 1,048,576 | 128 | 7 | 812.459 | 68.171 | 11.918x | 7–0 | 28.537 |
| 1,048,576 | 256 | 7 | 359.756 | 34.046 | 10.567x | 7–0 | 28.303 |
| 2,097,152 | 64 | 7 | 1,474.066 | 134.724 | 10.941x | 7–0 | 29.129 |
| 2,097,152 | 128 | 7 | 841.831 | 71.609 | 11.756x | 7–0 | 28.069 |
| 2,097,152 | 256 | 7 | 389.546 | 35.626 | 10.934x | 7–0 | 28.760 |
| 4,194,304 | 64 | 7 | 1,491.953 | 148.293 | 10.061x | 7–0 | 28.104 |
| 4,194,304 | 128 | 7 | 861.462 | 76.701 | 11.231x | 7–0 | 29.058 |
| 4,194,304 | 256 | 7 | 411.714 | 36.959 | 11.140x | 7–0 | 28.844 |
| 8,388,607 | 64 | 7 | 1,519.902 | 156.457 | 9.714x | 7–0 | 28.907 |
| 8,388,607 | 128 | 7 | 880.071 | 78.222 | 11.251x | 7–0 | 29.194 |
| 8,388,607 | 256 | 7 | 421.378 | 37.171 | 11.336x | 7–0 | 28.243 |

#### Extract

[PNG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-extract-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins | Regex IR JIT-ready mean (ms) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,024 | 64 | 3 | 6.685 | 2.902 | 2.303x | 3–0 | 45.096 |
| 1,024 | 128 | 3 | 6.666 | 2.550 | 2.614x | 3–0 | 44.241 |
| 1,024 | 256 | 3 | 6.738 | 2.263 | 2.977x | 3–0 | 42.407 |
| 2,048 | 64 | 3 | 13.493 | 5.590 | 2.414x | 3–0 | 41.179 |
| 2,048 | 128 | 3 | 13.522 | 4.994 | 2.708x | 3–0 | 41.571 |
| 2,048 | 256 | 3 | 13.474 | 4.643 | 2.902x | 3–0 | 42.487 |
| 4,096 | 64 | 3 | 27.337 | 10.838 | 2.522x | 3–0 | 42.559 |
| 4,096 | 128 | 3 | 26.078 | 9.558 | 2.728x | 3–0 | 41.977 |
| 4,096 | 256 | 3 | 22.560 | 8.079 | 2.793x | 3–0 | 41.915 |
| 8,192 | 64 | 3 | 50.321 | 19.783 | 2.544x | 3–0 | 41.320 |
| 8,192 | 128 | 3 | 43.006 | 16.712 | 2.573x | 3–0 | 42.600 |
| 8,192 | 256 | 3 | 44.692 | 17.087 | 2.616x | 3–0 | 41.993 |
| 16,384 | 64 | 3 | 88.335 | 32.235 | 2.740x | 3–0 | 41.277 |
| 16,384 | 128 | 3 | 67.557 | 32.179 | 2.099x | 3–0 | 41.230 |
| 16,384 | 256 | 3 | 67.451 | 29.014 | 2.325x | 3–0 | 42.379 |
| 32,768 | 64 | 3 | 111.076 | 49.577 | 2.240x | 3–0 | 43.483 |
| 32,768 | 128 | 3 | 110.781 | 44.104 | 2.512x | 3–0 | 51.655 |
| 32,768 | 256 | 3 | 111.453 | 40.255 | 2.769x | 3–0 | 43.769 |
| 65,536 | 64 | 3 | 132.041 | 54.031 | 2.444x | 3–0 | 41.804 |
| 65,536 | 128 | 3 | 127.452 | 44.839 | 2.842x | 3–0 | 44.352 |
| 65,536 | 256 | 3 | 127.344 | 39.299 | 3.240x | 3–0 | 41.215 |
| 131,072 | 64 | 3 | 157.960 | 61.169 | 2.582x | 3–0 | 42.163 |
| 131,072 | 128 | 3 | 152.603 | 52.691 | 2.896x | 3–0 | 42.362 |
| 131,072 | 256 | 3 | 149.377 | 46.654 | 3.202x | 3–0 | 41.571 |
| 262,144 | 64 | 3 | 184.473 | 63.991 | 2.883x | 3–0 | 48.631 |
| 262,144 | 128 | 3 | 177.135 | 55.220 | 3.208x | 3–0 | 42.772 |
| 262,144 | 256 | 3 | 171.994 | 49.708 | 3.460x | 3–0 | 41.991 |
| 524,288 | 64 | 3 | 193.025 | 71.869 | 2.686x | 3–0 | 40.740 |
| 524,288 | 128 | 3 | 184.748 | 61.264 | 3.016x | 3–0 | 45.243 |
| 524,288 | 256 | 3 | 174.986 | 54.683 | 3.200x | 3–0 | 43.270 |
| 1,048,576 | 64 | 3 | 222.340 | 76.610 | 2.902x | 3–0 | 44.062 |
| 1,048,576 | 128 | 3 | 220.356 | 64.995 | 3.390x | 3–0 | 43.357 |
| 1,048,576 | 256 | 3 | 215.161 | 58.046 | 3.707x | 3–0 | 41.167 |
| 2,097,152 | 64 | 3 | 265.821 | 108.799 | 2.443x | 3–0 | 42.669 |
| 2,097,152 | 128 | 3 | 262.656 | 84.958 | 3.092x | 3–0 | 42.757 |
| 2,097,152 | 256 | 3 | 256.201 | 73.231 | 3.499x | 3–0 | 45.801 |
| 4,194,304 | 64 | 3 | 386.867 | 132.056 | 2.930x | 3–0 | 41.432 |
| 4,194,304 | 128 | 3 | 361.510 | 105.592 | 3.424x | 3–0 | 41.709 |
| 4,194,304 | 256 | 3 | 338.700 | 86.386 | 3.921x | 3–0 | 42.674 |
| 8,388,607 | 64 | 3 | 314.780 | 139.200 | 2.261x | 3–0 | 41.975 |
| 8,388,607 | 128 | 3 | 420.291 | 94.498 | 4.448x | 3–0 | 57.843 |
| 8,388,607 | 256 | 3 | 393.992 | 84.997 | 4.635x | 3–0 | 42.931 |

#### Replace

[PNG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-replace-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins | Regex IR JIT-ready mean (ms) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,024 | 64 | 14 | 7.258 | 0.894 | 8.115x | 14–0 | 62.277 |
| 1,024 | 128 | 14 | 4.402 | 0.470 | 9.364x | 14–0 | 64.582 |
| 1,024 | 256 | 14 | 2.447 | 0.240 | 10.198x | 14–0 | 60.613 |
| 2,048 | 64 | 14 | 13.903 | 1.734 | 8.018x | 14–0 | 61.195 |
| 2,048 | 128 | 14 | 8.530 | 0.909 | 9.383x | 14–0 | 63.531 |
| 2,048 | 256 | 14 | 4.619 | 0.443 | 10.419x | 14–0 | 63.340 |
| 4,096 | 64 | 14 | 27.439 | 3.410 | 8.047x | 14–0 | 62.069 |
| 4,096 | 128 | 14 | 16.448 | 1.782 | 9.228x | 14–0 | 64.914 |
| 4,096 | 256 | 14 | 8.742 | 0.895 | 9.767x | 14–0 | 62.587 |
| 8,192 | 64 | 14 | 52.260 | 6.514 | 8.023x | 14–0 | 62.363 |
| 8,192 | 128 | 14 | 30.359 | 3.482 | 8.718x | 14–0 | 62.682 |
| 8,192 | 256 | 14 | 16.706 | 1.731 | 9.650x | 14–0 | 65.617 |
| 16,384 | 64 | 14 | 101.335 | 12.071 | 8.395x | 14–0 | 61.710 |
| 16,384 | 128 | 14 | 54.410 | 6.291 | 8.649x | 14–0 | 60.890 |
| 16,384 | 256 | 14 | 31.115 | 3.291 | 9.456x | 14–0 | 62.370 |
| 32,768 | 64 | 14 | 150.025 | 19.129 | 7.843x | 14–0 | 61.621 |
| 32,768 | 128 | 14 | 93.671 | 10.071 | 9.301x | 14–0 | 63.898 |
| 32,768 | 256 | 14 | 53.377 | 4.998 | 10.680x | 14–0 | 62.777 |
| 65,536 | 64 | 14 | 235.879 | 23.031 | 10.242x | 14–0 | 60.993 |
| 65,536 | 128 | 14 | 139.199 | 11.547 | 12.055x | 14–0 | 63.231 |
| 65,536 | 256 | 14 | 72.597 | 5.590 | 12.987x | 14–0 | 68.591 |
| 131,072 | 64 | 14 | 317.706 | 28.203 | 11.265x | 14–0 | 66.514 |
| 131,072 | 128 | 14 | 162.680 | 14.052 | 11.577x | 14–0 | 69.420 |
| 131,072 | 256 | 14 | 57.202 | 6.707 | 8.529x | 14–0 | 62.273 |
| 262,144 | 64 | 14 | 296.474 | 31.379 | 9.448x | 14–0 | 61.829 |
| 262,144 | 128 | 14 | 155.315 | 15.940 | 9.744x | 14–0 | 60.878 |
| 262,144 | 256 | 14 | 59.174 | 7.555 | 7.832x | 14–0 | 61.542 |
| 524,288 | 64 | 14 | 332.968 | 34.393 | 9.681x | 14–0 | 62.693 |
| 524,288 | 128 | 14 | 174.011 | 17.292 | 10.063x | 14–0 | 60.510 |
| 524,288 | 256 | 14 | 74.113 | 8.215 | 9.021x | 14–0 | 69.412 |
| 1,048,576 | 64 | 14 | 368.784 | 36.895 | 9.995x | 14–0 | 62.928 |
| 1,048,576 | 128 | 14 | 210.388 | 18.405 | 11.431x | 14–0 | 61.490 |
| 1,048,576 | 256 | 14 | 88.278 | 8.690 | 10.159x | 14–0 | 62.633 |
| 2,097,152 | 64 | 14 | 441.060 | 39.161 | 11.263x | 14–0 | 65.523 |
| 2,097,152 | 128 | 14 | 240.072 | 19.175 | 12.520x | 14–0 | 62.190 |
| 2,097,152 | 256 | 14 | 97.101 | 8.965 | 10.831x | 14–0 | 60.558 |
| 4,194,304 | 64 | 14 | 496.519 | 40.227 | 12.343x | 14–0 | 61.834 |
| 4,194,304 | 128 | 14 | 259.205 | 19.518 | 13.280x | 14–0 | 62.106 |
| 4,194,304 | 256 | 14 | 102.337 | 9.076 | 11.275x | 14–0 | 62.635 |
| 8,388,607 | 64 | 14 | 545.605 | 40.449 | 13.489x | 14–0 | 64.113 |
| 8,388,607 | 128 | 14 | 270.534 | 19.316 | 14.005x | 14–0 | 62.277 |
| 8,388,607 | 256 | 14 | 105.647 | 8.922 | 11.842x | 14–0 | 60.144 |

#### Split

[PNG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-split-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins | Regex IR JIT-ready mean (ms) |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,024 | 64 | 7 | 4.610 | 1.087 | 4.241x | 7–0 | 60.185 |
| 1,024 | 128 | 7 | 3.295 | 0.607 | 5.427x | 7–0 | 58.695 |
| 1,024 | 256 | 7 | 2.144 | 0.323 | 6.633x | 7–0 | 58.554 |
| 2,048 | 64 | 7 | 8.991 | 2.126 | 4.230x | 7–0 | 63.422 |
| 2,048 | 128 | 7 | 6.218 | 1.203 | 5.169x | 7–0 | 59.718 |
| 2,048 | 256 | 7 | 3.954 | 0.615 | 6.427x | 7–0 | 58.655 |
| 4,096 | 64 | 7 | 16.054 | 4.213 | 3.811x | 7–0 | 59.766 |
| 4,096 | 128 | 7 | 11.749 | 2.340 | 5.021x | 7–0 | 58.918 |
| 4,096 | 256 | 7 | 7.342 | 1.226 | 5.990x | 7–0 | 57.829 |
| 8,192 | 64 | 7 | 28.773 | 7.902 | 3.641x | 7–0 | 60.689 |
| 8,192 | 128 | 7 | 21.250 | 4.473 | 4.750x | 7–0 | 57.565 |
| 8,192 | 256 | 7 | 13.453 | 2.371 | 5.674x | 7–0 | 58.027 |
| 16,384 | 64 | 7 | 50.638 | 13.723 | 3.690x | 7–0 | 59.267 |
| 16,384 | 128 | 7 | 32.278 | 7.891 | 4.091x | 7–0 | 57.490 |
| 16,384 | 256 | 7 | 21.096 | 4.332 | 4.870x | 7–0 | 59.273 |
| 32,768 | 64 | 7 | 71.883 | 20.419 | 3.520x | 7–0 | 58.009 |
| 32,768 | 128 | 7 | 47.404 | 12.596 | 3.763x | 7–0 | 57.876 |
| 32,768 | 256 | 7 | 29.466 | 6.675 | 4.415x | 7–0 | 59.510 |
| 65,536 | 64 | 7 | 93.118 | 26.499 | 3.514x | 7–0 | 62.003 |
| 65,536 | 128 | 7 | 56.366 | 15.158 | 3.719x | 7–0 | 58.872 |
| 65,536 | 256 | 7 | 35.107 | 8.160 | 4.303x | 7–0 | 58.737 |
| 131,072 | 64 | 7 | 106.504 | 30.919 | 3.445x | 7–0 | 58.891 |
| 131,072 | 128 | 7 | 68.933 | 18.296 | 3.768x | 7–0 | 59.458 |
| 131,072 | 256 | 7 | 40.102 | 9.646 | 4.157x | 7–0 | 59.107 |
| 262,144 | 64 | 7 | 109.496 | 34.167 | 3.205x | 7–0 | 59.236 |
| 262,144 | 128 | 7 | 76.543 | 20.338 | 3.763x | 7–0 | 59.791 |
| 262,144 | 256 | 7 | 44.095 | 10.653 | 4.139x | 7–0 | 59.042 |
| 524,288 | 64 | 7 | 121.256 | 39.960 | 3.034x | 7–0 | 72.834 |
| 524,288 | 128 | 7 | 88.553 | 22.853 | 3.875x | 7–0 | 60.273 |
| 524,288 | 256 | 7 | 55.939 | 12.055 | 4.640x | 7–0 | 58.306 |
| 1,048,576 | 64 | 7 | 155.042 | 44.617 | 3.475x | 7–0 | 58.822 |
| 1,048,576 | 128 | 7 | 116.548 | 24.967 | 4.668x | 7–0 | 58.990 |
| 1,048,576 | 256 | 7 | 71.631 | 12.826 | 5.585x | 7–0 | 59.737 |
| 2,097,152 | 64 | 7 | 205.644 | 49.199 | 4.180x | 7–0 | 57.986 |
| 2,097,152 | 128 | 7 | 134.118 | 27.013 | 4.965x | 7–0 | 57.845 |
| 2,097,152 | 256 | 7 | 83.298 | 13.600 | 6.125x | 7–0 | 57.831 |
| 4,194,304 | 64 | 7 | 226.523 | 54.991 | 4.119x | 7–0 | 57.884 |
| 4,194,304 | 128 | 7 | 162.296 | 28.957 | 5.605x | 7–0 | 60.125 |
| 4,194,304 | 256 | 7 | 95.074 | 14.136 | 6.726x | 7–0 | 57.950 |
| 8,388,607 | 64 | 7 | 311.022 | 58.587 | 5.309x | 7–0 | 59.346 |
| 8,388,607 | 128 | 7 | 183.366 | 29.718 | 6.170x | 7–0 | 60.170 |
| 8,388,607 | 256 | 7 | 70.054 | 14.147 | 4.952x | 7–0 | 59.285 |

[All API measurements](docs/_static/benchmarks/cudf-api-throughput-data.csv)

<!-- END GENERATED CUDF API RESULTS -->

The `regex_ir/cold` and `cudf/cold` registrations remain available for
one-shot setup-plus-first-execution measurements. Their separate `JIT Ready`
summary stops after module loading and function lookup, before output
allocation or the first launch. Production integrations should cache the
linked cubin by pattern, options, architecture, and toolkit version.

At one million 128-byte rows, the current warm and cold measurements are:

| Pattern | Regex IR warm (ms) | cuDF warm (ms) | Warm speedup | Regex IR JIT-ready (ms) | Regex IR cold JIT + first launch (ms) | cuDF program + first call (ms) |
|:---|---:|---:|---:|---:|---:|---:|
| log | 1.370 | 2.090 | 1.525x | 26.847 | 25.255 | 2.088 |
| email | 1.452 | 9.396 | 6.469x | 23.694 | 25.009 | 9.466 |
| URI | 1.455 | 10.802 | 7.426x | 23.937 | 24.986 | 10.942 |
| IPv4 | 1.518 | 28.391 | 18.698x | 24.335 | 25.562 | 28.471 |

The log, email, and URI cold paths do not repay roughly 23–24 ms of device
compilation on the first call. IPv4 repays that cost narrowly at one million
rows. Cached use avoids it entirely.

### Suites B–E results: imported complete corpora

The generated block below is the source-of-truth table for all 74 expressions
that scan an upstream source corpus. It reports normal execution and uncached
JIT-ready time separately. The 14 Boost scalar records are useful launch and
exact-row controls rather than corpus scans, so they are summarized separately:

| Workload | Cases | Regex IR mean (ms) | cuDF mean (ms) | Geometric speedup | Pair wins | Regex IR JIT-ready mean (ms) | cuDF program-create mean (ms) |
|:---|---:|---:|---:|---:|:---:|---:|---:|
| Boost repeated scalar records | 14 | 0.071 | 3.465 | 48.672x | 14–0 | 22.977 | 0.0104 |

All 28 scalar engine states completed and passed the same pre-timing output
comparison. Regex IR JIT-ready includes parse/lower, NVVM rendering, libNVVM
`-opt=3 -gen-lto`, cache-disabled nvJitLink device LTO at `-O3`, module load,
and function lookup. cuDF program creation does not JIT a specialized kernel.

<!-- BEGIN GENERATED CORPUS SWEEP RESULTS -->

#### Complete-corpus case charts

The linked presentation charts use uncluttered grouped horizontal bars for every
imported suite without embedding images in this README. Each point is one upstream
regex case over its complete source corpus; throughput uses input bytes rather than
row count so differently packed cuDF columns remain comparable. The 14 Boost scalar
records remain in the suite summary above but are intentionally absent here because
they are repeated literals rather than source corpora.
Panels spanning at least 50x use a marked logarithmic throughput axis; narrower
panels retain a linear axis.

These cases were rerun on 2026-07-07 with at least five samples, 0.05 seconds of
measured GPU time, a 2% target-noise threshold, and a 10-second per-state timeout.
All 148 engine states completed without warnings, skips, or timeouts, and every
pre-timing Regex IR/cuDF output comparison passed.
JIT-ready time is uncached and spans the regex string through loaded module and
resolved kernel function; corpus setup and the first launch are excluded.

| Source suite | Corpus expressions | Regex IR geometric throughput (GiB/s) | cuDF geometric throughput (GiB/s) | Geometric speedup | Pair wins | Regex IR JIT-ready mean (ms) | cuDF program-create mean (ms) |
|:---|---:|---:|---:|---:|:---:|---:|---:|
| OpenResty | 31 | 54.831 | 2.791 | 19.649x | 31–0 | 24.201 | 0.0151 |
| Rust Leipzig | 18 | 42.712 | 2.031 | 21.033x | 18–0 | 30.074 | 0.0219 |
| Boost/GCC | 22 | 2.890 | 0.267 | 10.827x | 22–0 | 39.243 | 0.0567 |
| mariomka | 3 | 24.306 | 1.014 | 23.964x | 3–0 | 24.655 | 0.0094 |
| **All complete-corpus suites** | **74** | **20.813** | **1.234** | **16.868x** | **74–0** | **30.120** | **0.0289** |

Regex IR won 74 of 74 paired full-corpus cases. The narrowest result was Boost/GCC case
8 (2.392x); the largest was Boost/GCC case 14 (690.046x).

#### OpenResty full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-openresty-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup | Regex IR JIT-ready (ms) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | literal miss | 25.000 | 0.282 | 1.107 | 3.917x | 10.117 |
| 2 | short alt miss | 10.000 | 0.139 | 2.527 | 18.151x | 22.720 |
| 3 | suffix alt miss | 25.000 | 0.389 | 7.930 | 20.386x | 22.174 |
| 4 | suffix alt prose | 19.117 | 0.410 | 8.110 | 19.800x | 23.142 |
| 5 | wide class miss | 25.000 | 0.335 | 3.520 | 10.511x | 22.346 |
| 6 | split class miss | 25.000 | 0.340 | 3.602 | 10.579x | 22.028 |
| 7 | split class prose | 19.117 | 0.202 | 2.365 | 11.709x | 23.255 |
| 8 | large alt prose | 19.117 | 0.452 | 34.395 | 76.099x | 23.789 |
| 9 | large alt miss | 25.000 | 0.384 | 32.908 | 85.785x | 22.851 |
| 10 | nested alt | 25.000 | 0.380 | 7.430 | 19.539x | 21.699 |
| 11 | long nested alt miss | 10.000 | 0.181 | 8.494 | 46.870x | 21.508 |
| 12 | capture chain miss | 25.000 | 0.379 | 8.245 | 21.783x | 21.950 |
| 13 | capture chain random miss | 10.000 | 0.182 | 8.408 | 46.174x | 21.531 |
| 14 | lazy class repeat | 10.000 | 0.138 | 1.366 | 9.871x | 22.294 |
| 15 | lazy dot repeat | 10.000 | 0.135 | 1.375 | 10.185x | 22.156 |
| 16 | greedy dot repeat | 10.000 | 0.137 | 1.375 | 10.044x | 22.268 |
| 17 | anchored literal | 19.117 | 0.403 | 3.853 | 9.565x | 24.285 |
| 18 | literal prose | 19.117 | 0.273 | 1.779 | 6.508x | 25.504 |
| 19 | folded literal | 19.117 | 0.387 | 4.202 | 10.861x | 24.360 |
| 20 | class suffix | 19.117 | 0.411 | 5.557 | 13.535x | 23.757 |
| 21 | name alternation | 19.117 | 0.397 | 5.657 | 14.232x | 28.984 |
| 22 | word boundary | 19.117 | 0.446 | 8.250 | 18.513x | 24.933 |
| 23 | negated bounded | 19.117 | 0.356 | 13.938 | 39.109x | 47.288 |
| 24 | name literals | 19.117 | 0.431 | 8.759 | 20.326x | 23.666 |
| 25 | folded names | 19.117 | 0.487 | 10.457 | 21.467x | 23.245 |
| 26 | short prefix names | 19.117 | 0.434 | 25.383 | 58.522x | 23.464 |
| 27 | required prefix names | 19.117 | 0.439 | 28.554 | 65.103x | 28.663 |
| 28 | word suffix | 19.117 | 0.368 | 6.856 | 18.624x | 24.367 |
| 29 | bounded word suffix | 19.117 | 0.473 | 11.238 | 23.746x | 27.906 |
| 30 | captured name suffix | 19.117 | 0.445 | 9.610 | 21.603x | 27.422 |
| 31 | quoted sentence | 19.117 | 0.403 | 14.705 | 36.492x | 26.546 |

#### Rust Leipzig full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-leipzig-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup | Regex IR JIT-ready (ms) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | twain | 15.272 | 0.215 | 1.636 | 7.623x | 30.220 |
| 2 | twain ignore case | 15.272 | 0.313 | 3.680 | 11.765x | 22.076 |
| 3 | shing | 15.272 | 0.336 | 4.755 | 14.141x | 21.712 |
| 4 | huck saw | 15.272 | 0.322 | 4.669 | 14.515x | 21.759 |
| 5 | word nn | 15.272 | 0.361 | 7.000 | 19.373x | 23.839 |
| 6 | negated bounded | 15.272 | 0.285 | 11.408 | 39.990x | 47.790 |
| 7 | names | 15.272 | 0.353 | 7.670 | 21.729x | 22.160 |
| 8 | names ignore case | 15.272 | 0.401 | 9.005 | 22.449x | 21.976 |
| 9 | optional prefix | 15.272 | 0.356 | 21.776 | 61.239x | 25.013 |
| 10 | required prefix | 15.272 | 0.357 | 21.678 | 60.727x | 22.575 |
| 11 | tom river | 15.272 | 1.365 | 13.404 | 9.819x | 115.623 |
| 12 | word ing | 15.272 | 0.305 | 5.515 | 18.101x | 22.428 |
| 13 | bounded ing | 15.272 | 0.389 | 10.445 | 26.873x | 24.205 |
| 14 | name suffix | 15.272 | 0.363 | 7.667 | 21.149x | 22.626 |
| 15 | quoted sentence | 15.272 | 0.328 | 13.176 | 40.227x | 22.344 |
| 16 | unicode symbols | 15.272 | 0.274 | 3.601 | 13.153x | 22.061 |
| 17 | math symbol property | 15.272 | 0.278 | 2.781 | 9.997x | 30.076 |
| 18 | csv field | 15.272 | 0.303 | 14.210 | 46.937x | 22.849 |

#### Boost/GCC full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-boost-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup | Regex IR JIT-ready (ms) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | long twain | 19.117 | 0.271 | 1.801 | 6.642x | 24.325 |
| 2 | long huck | 19.117 | 0.270 | 2.052 | 7.612x | 28.768 |
| 3 | long ing | 19.117 | 0.369 | 6.909 | 18.723x | 23.456 |
| 4 | long line twain | 19.117 | 0.406 | 7.097 | 17.482x | 26.068 |
| 5 | long names | 19.117 | 0.434 | 8.574 | 19.760x | 24.695 |
| 6 | long names near river | 19.117 | 2.007 | 27.750 | 13.825x | 83.401 |
| 7 | medium twain | 0.048 | 0.032 | 0.088 | 2.743x | 25.022 |
| 8 | medium huck | 0.048 | 0.032 | 0.078 | 2.392x | 22.532 |
| 9 | medium ing | 0.048 | 0.033 | 0.348 | 10.529x | 21.682 |
| 10 | medium line twain | 0.048 | 0.034 | 0.346 | 10.043x | 22.652 |
| 11 | medium names | 0.048 | 0.036 | 0.290 | 8.169x | 21.864 |
| 12 | medium names near river | 0.048 | 0.127 | 0.629 | 4.945x | 79.703 |
| 13 | cpp declaration | 0.033 | 0.034 | 0.342 | 10.096x | 82.483 |
| 14 | cpp tokens | 0.033 | 0.120 | 82.822 | 690.046x | 178.944 |
| 15 | cpp include | 0.033 | 0.033 | 0.234 | 7.146x | 26.939 |
| 16 | boost include | 0.033 | 0.034 | 0.239 | 7.023x | 28.825 |
| 17 | html names | 0.049 | 0.039 | 0.288 | 7.419x | 22.745 |
| 18 | html paragraph | 0.049 | 0.032 | 0.284 | 8.871x | 22.560 |
| 19 | html anchor | 0.049 | 0.037 | 0.789 | 21.264x | 23.603 |
| 20 | html heading | 0.049 | 0.032 | 0.306 | 9.480x | 23.551 |
| 21 | html image | 0.049 | 0.033 | 0.426 | 12.780x | 23.016 |
| 22 | html font | 0.049 | 0.036 | 0.278 | 7.643x | 26.511 |

#### mariomka full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-mariomka-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup | Regex IR JIT-ready (ms) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | email | 6.523 | 0.258 | 3.585 | 13.904x | 27.900 |
| 2 | uri | 6.523 | 0.267 | 6.907 | 25.904x | 23.323 |
| 3 | ipv4 | 6.523 | 0.262 | 10.003 | 38.211x | 22.743 |

[All complete-corpus measurements](docs/_static/benchmarks/corpus-throughput-data.csv)

<!-- END GENERATED CORPUS SWEEP RESULTS -->

The previous assertion outliers are gone. Boolean word boundaries, line
anchors, end assertions, and CRLF-sensitive assertions now use an
assertion-aware deterministic table instead of falling back to recursive
Thompson execution. The full-corpus word-boundary and FTP expressions are
included in the 74–0 source-corpus result rather than excluded or hidden.

The imported workloads also drove two code-generation changes. Large boolean
alternations are split into independently optimized DFA functions and joined
by a short-circuit wrapper: the Leipzig bounded `Tom...river` case fell from
about 50 seconds historically to 1.365 ms here, versus cuDF at 13.404 ms.
Boolean machines reclaim the unused priority flag as a fifteenth state bit;
the 16,385-state bounded negated-class case was reduced from 94–109 ms by the
DFA and now avoids that table entirely through the gated Glushkov plan. It runs
in 0.361 ms here, versus cuDF at 13.938 ms. Transition tables that still exceed
the 64 KiB constant segment are emitted into read-only global storage.

### Profile-guided optimization

The toolchain already used libNVVM `-opt=3`; it now also passes nvJitLink
`-lto -O3`. Nsight Systems showed that two baseline count launches averaged
10.027 ms each while module load and launch APIs were tiny, proving that the
slowdown was in device execution rather than host JIT setup. Cubin SASS showed
fourteen register saves and restores around recursive character transitions.
That call-frame traffic, repeated UTF-8 checks, and warp-divergent backtracking
turned a small input into gigabytes of local-memory traffic.

The generated NVVM now selects among specialized executor strategies:

- direct exact-ASCII compares/finders, including a single-byte scan, packed
  8/4/2/1-byte verification, and an eight-candidate long-literal scan;
- an existential DFA for boolean contains/match operations, where
  greedy/lazy priority is unobservable and the reclaimed flag bit permits up
  to 32,767 encoded states;
- a gated 64-bit Glushkov NFA for assertion-free, non-nullable boolean graphs
  where sparse position shifts remove a large DFA table or a long linear
  follow graph wins the measured cost model;
- an ordered DFA for capture-free span/global operations, preserving
  alternation and greedy/lazy priority with ordered state sets and a
  stop-before-accept bit;
- a tagged DFA for capture graphs proven to have one consuming thread per
  alphabet class and terminal acceptance;
- the recursive Thompson executor as the general correctness fallback for
  non-boolean internal assertions and ambiguous capture histories.

Ordered machines additionally use sparse start-range filtering and can skip a
failed self-looping prefix run when the automaton proves every skipped start
would reach the same failure. Small byte alphabets use inline range
classification instead of a constant-table load when that reduces the loop.

Large top-level boolean alternations can be partitioned into smaller DFA
functions with short-circuit dispatch. Transition tables larger than 32 KiB
use read-only global storage, preventing multiple split tables from exceeding
the device's 64 KiB constant-data limit.

Global beginning/end anchors are folded into deterministic control flow, while
the generated assertion helper validates end-line semantics. A replacement
capture proven to cover the complete match reuses the match span instead of
materializing a capture array. Ordinary loads retain compiler-selected cache
policy. The large API benchmark wrappers use 256-thread blocks; the smaller
imported-corpus grids use a profile-selected 128-thread launch. Extract writes
captures directly to final device storage.

Nsight Compute used the same 262,144-row, 128-byte-axis workloads:

| Count pattern 0 metric | Recursive baseline | Prioritized DFA | Change |
|:---|---:|---:|---:|
| Kernel duration | 9.940 ms | 0.688 ms | 14.4x faster |
| Executed instructions | 663.5 million | 208.0 million | 3.2x fewer |
| Warp cycles per issued instruction | 32.49 | 9.31 | 3.5x lower |
| Long-scoreboard stall cycles/instruction | 19.43 | 1.52 | 12.8x lower |
| Measured memory throughput | 465.6 GB/s | 30.8 GB/s | call-stack traffic removed |
| Achieved occupancy | 30.86% | 43.90% | +13.04 points |

| Four-group extract metric | Recursive baseline | Tagged DFA | Change |
|:---|---:|---:|---:|
| Kernel duration | 14.880 ms | 0.266 ms | 56.0x faster |
| Executed instructions | 207.9 million | 8.3 million | 25.0x fewer |
| Average active threads per warp | 7.81 | 27.85 | 3.6x higher |
| LG-throttle stall cycles/instruction | 69.41 | 11.15 | 6.2x lower |
| Registers per thread | 94 | 78 | 17.0% fewer |
| Achieved occupancy | 29.96% | 45.31% | +15.35 points |

The 14-point API sweep exposed two additional gaps that the earlier
262K/2M-only sampling obscured. Literal-space count, replacement, and split
were using the full prioritized DFA even though their complete regex is one
ASCII byte. Begin-anchored boolean machines also kept processing a dead state
to the end of every nonmatching row. The generator now emits a direct byte
search for the former and immediately rejects dead non-scanning DFA states for
the latter.

Full NCU replay on split pattern 1 at 2,097,152 rows and 256 bytes measured:

| Split kernel metric | Generic prioritized DFA | Direct byte scan | Change |
|:---|---:|---:|---:|
| Size-kernel duration | 11.11 ms | 2.22 ms | 5.0x faster |
| Emit-kernel duration | 11.22 ms | 3.78 ms | 3.0x faster |
| Size executed instructions | 2.685 billion | 330.3 million | 8.1x fewer |
| Emit executed instructions | 2.695 billion | 340.4 million | 7.9x fewer |
| Registers per thread | 84 | 38 | 54.8% fewer |
| Size achieved occupancy | 29.96% | 70.46% | +40.50 points |
| Emit achieved occupancy | 29.96% | 72.12% | +42.16 points |
| Average active threads/warp | 6.5 | 16.7 | 2.6x higher |

At the 8,388,607-row/256-byte endpoint, normal NVBench execution moved
literal-space count from 31.57 ms to 7.94 ms, replacement from 98.81 ms to
42.96 ms, and split from 97.21 ms to 34.68 ms. The corresponding cuDF times
were 28.22, 80.83, and 73.40 ms. For the previous worst anchored-contains
case, dead-state rejection reduced Regex IR from 72.83 us to 10.78 us versus
cuDF at 53.72 us.

After refreshing every affected geometry, Regex IR wins all 1,764 paired API
measurements. The remaining generic-DFA opportunity is still instruction and
lane efficiency on complex patterns; the direct-byte profile is now
memory-throughput limited rather than register/occupancy limited. Variable-size
replacement and split necessarily retain separate sizing and emission passes
unless an integration accepts over-allocation or persistent span scratch.

A subsequent highest-value-opportunity PGO pass used the existing
2,097,152-row, 128-byte-axis gate. Relative to its pre-pass baseline, geometric
mean latency improved 5.3% for contains, 22.2% for count, 28.1% for replace,
and 25.3% for split; extract was unchanged within 0.1%. No individual stable
state regressed by more than 3%. The complete 58-case large-corpus gate
improved by 7.0% geometrically. The largest API case, `[a-z]+Z`, improved from
6.828 to 3.689 ms for count, 20.493 to 11.121 ms for plain replacement, and
19.895 to 12.930 ms for split.

The 2026-07-07 all-benchmark PGO pass then replayed all 1,764 Regex IR API
states, all 88 imported-corpus states, and all warm/cold states from a frozen
branch-point baseline. A sparse ASCII start-filter miss in count now advances
by the already-known one-byte width instead of loading that byte again. The
complete count matrix improved 2.7%; a controlled large-state ABBA gate
measured 12.6% for `\d+` and 9.3% for `[a-f]+|[0-5]+`. NCU measured 12.8%
fewer executed instructions and 31.5% fewer global-load instructions for the
profiled `\d+` kernel, with unchanged registers and occupancy. The direct edge
is count-only: applying it to the larger replacement CFG slowed large rows by
5–8%, so replace, split, find, and extract retain their prior lowering.

The corpus-only 128-thread launch improved every imported family in ABBA
measurements and raised the complete-corpus geometric speedup from 16.606x to
16.868x. Broader `alwaysinline`, 32-byte start-filter, unfiltered-dead-edge,
and universal 128-thread experiments were rejected after neutral or regressive
profiles.

[optimization.md](optimization.md) records the full experiment matrix,
including rejected staged materialization, selectivity sweeps, cached
length-ordering and pivot upper bounds, the temporary multi-pattern workload,
and Nsight Compute metrics.

The final complete-corpus build was also replayed with NCU's full metric set.
Each row below is one of the eight 32,768-row column launches; NCU replay
overhead is why these kernel durations are not substituted into the NVBench
tables.

| Complete-corpus kernel | Kernel duration | Instructions | Registers/thread | Branch efficiency | Achieved occupancy | Spill traffic |
|:---|---:|---:|---:|---:|---:|---:|
| OpenResty case 22, word boundary | 57.95 us | 7.92 million | 77 | 96.80% | 23.62% | 0 B |
| mariomka case 2, URI | 35.20 us | 3.10 million | 75 | 90.85% | 21.91% | 0 B |

Both kernels are compute-limited rather than bandwidth-limited: NCU measured
45.36% and 30.44% compute-pipe utilization, versus 6.81% and 4.64% DRAM
utilization. The remaining opportunity is reducing classification/transition
instructions and divergent end-of-row control, not forcing a cache policy or
trading registers for occupancy. Neither profile contains recursive call-frame
spills.

Profiler replay perturbs absolute latency, so the benchmark table uses normal
NVBench timings. Set `REGEX_IR_BENCHMARK_DUMP_DIR` to retain generated
`.nvvm.ll`, libNVVM `.ltoir`, and linked `.cubin` files for `cuobjdump` or
`nvdisasm` inspection.

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

The separately reported JIT-ready latency includes parsing, determinization,
libNVVM LTO-IR generation, device LTO, module loading, and function lookup, but
not the first launch. cuDF copies a compact predefined instruction program
instead of invoking a device compiler, so it remains faster for uncached
one-shot use. Production users should cache the linked cubin by pattern,
options, architecture, and toolkit version; steady-state execution is the path
where JIT specialization pays off.

## Test strategy

The default test suite includes:

- the host semantic interpreter for all operation result shapes;
- 190 boolean cases: 70 project scenarios, 20 adapted from RE2, 53 from Rust's
  `regex` crate, 26 from CPython `re_tests`, and 21 from sihlfall's exhaustive
  RE2-derived categories;
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
`Project`, `Re2`, `RustRegex`, `CPython`, `Sihlfall`, `Cudf`, and `Nvvm`
fixtures. Every one of the 92 cuDF test names has its own `TEST_F(Cudf, ...)`
entry. Stable device entry points live in the [test fragments](tests/fragments):
boolean, find, count, capture, replace, and split each have a named CUDA source
and are embedded as NVCC-generated LTO-IR fatbins without macro-selected source
variants. License details are in
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
- ASCII POSIX bracket classes and the Unicode mathematical-symbol property
  `\p{Sm}`/`\P{Sm}`;
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
