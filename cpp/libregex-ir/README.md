# Regex IR

## What this project is about

Regex IR is a small, independent C++20 compiler for turning regular expressions
into inspectable intermediate representations and NVVM IR.
It is aimed at GPU systems that want to specialize a matcher for one pattern,
link that matcher into an existing CUDA kernel, and avoid interpreting generic
regex opcodes at runtime.

The project owns regex parsing, ordered Thompson construction, typed lowering,
optimization, verification, IR printing, prioritized and tagged deterministic
specialization, and textual NVVM IR generation. NVVM IR uses LLVM-derived
syntax but is NVIDIA's restricted, CUDA-specific device IR; the public API
intentionally does not claim to emit generic LLVM IR. The core library has no
LLVM, CUDA, or cuDF dependency: it assembles NVVM IR with ordinary strings and
libfmt's `fmt::format`. A consumer passes that string to libNVVM with
`-gen-lto`, links the resulting LTO-IR to its kernel LTO-IR with nvJitLink, then
launches the cubin through its existing CUDA runtime.

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
      -> prioritized DFA, tagged DFA, or ordered Thompson fallback
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
nvJitLink combines them with generated operation-specific PTX. The same
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
behavior. Capture-free programs use ordered deterministic states when the
state-table resource limits permit it. Capture programs use tagged
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

The complete cuDF-derived matrix was rerun on 2026-07-05 with the current code.
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

| API | Expressions | Geometries | Paired measurements | Regex IR geometric throughput (M rows/s) | cuDF geometric throughput (M rows/s) | Geometric speedup | Pair wins |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| contains | 11 | 42 | 462 | 680.059 | 78.176 | 8.699x | 462–0 |
| count | 7 | 42 | 294 | 175.950 | 25.139 | 6.999x | 294–0 |
| extract | 3 | 42 | 126 | 107.814 | 33.819 | 3.188x | 126–0 |
| replace | 14 | 42 | 588 | 63.182 | 7.159 | 8.826x | 588–0 |
| split | 7 | 42 | 294 | 40.309 | 9.407 | 4.285x | 294–0 |
| **All APIs** | **42** | **42** | **1764** | **134.600** | **19.302** | **6.973x** | **1764–0** |

Regex IR won 1764 of 1764 paired measurements. Contains, count, extract, replace, and
split won every state. The narrowest result was count pattern 1 at 131,072 rows/256
bytes (1.083x), while the largest was contains pattern 2 at 8,388,607 rows/64 bytes
(50.368x). The results describe this hardware, corpus, and allocation contract—not every
regex or deployment. Use the commands above to reproduce the matrix.

The tables below report geometric mean throughput across each API's expression
cases independently at every row-count/`StringBytes` geometry.

#### Contains

[PNG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-contains-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 11 | 51.907 | 9.014 | 5.759x | 11–0 |
| 1,024 | 128 | 11 | 30.974 | 5.939 | 5.215x | 11–0 |
| 1,024 | 256 | 11 | 20.586 | 3.778 | 5.449x | 11–0 |
| 2,048 | 64 | 11 | 102.663 | 17.994 | 5.705x | 11–0 |
| 2,048 | 128 | 11 | 61.364 | 11.835 | 5.185x | 11–0 |
| 2,048 | 256 | 11 | 40.606 | 7.524 | 5.397x | 11–0 |
| 4,096 | 64 | 11 | 200.130 | 35.935 | 5.569x | 11–0 |
| 4,096 | 128 | 11 | 121.761 | 23.522 | 5.176x | 11–0 |
| 4,096 | 256 | 11 | 80.814 | 14.872 | 5.434x | 11–0 |
| 8,192 | 64 | 11 | 390.872 | 60.730 | 6.436x | 11–0 |
| 8,192 | 128 | 11 | 235.601 | 35.811 | 6.579x | 11–0 |
| 8,192 | 256 | 11 | 159.363 | 23.458 | 6.794x | 11–0 |
| 16,384 | 64 | 11 | 734.044 | 84.370 | 8.700x | 11–0 |
| 16,384 | 128 | 11 | 464.846 | 61.709 | 7.533x | 11–0 |
| 16,384 | 256 | 11 | 315.997 | 43.528 | 7.260x | 11–0 |
| 32,768 | 64 | 11 | 1,411.249 | 119.304 | 11.829x | 11–0 |
| 32,768 | 128 | 11 | 863.081 | 90.540 | 9.533x | 11–0 |
| 32,768 | 256 | 11 | 568.413 | 63.347 | 8.973x | 11–0 |
| 65,536 | 64 | 11 | 1,773.936 | 145.886 | 12.160x | 11–0 |
| 65,536 | 128 | 11 | 1,033.443 | 110.062 | 9.390x | 11–0 |
| 65,536 | 256 | 11 | 658.282 | 80.720 | 8.155x | 11–0 |
| 131,072 | 64 | 11 | 2,512.586 | 160.065 | 15.697x | 11–0 |
| 131,072 | 128 | 11 | 1,437.269 | 116.328 | 12.355x | 11–0 |
| 131,072 | 256 | 11 | 890.868 | 85.210 | 10.455x | 11–0 |
| 262,144 | 64 | 11 | 3,207.143 | 178.405 | 17.977x | 11–0 |
| 262,144 | 128 | 11 | 1,806.172 | 129.664 | 13.930x | 11–0 |
| 262,144 | 256 | 11 | 1,099.634 | 94.882 | 11.589x | 11–0 |
| 524,288 | 64 | 11 | 3,830.750 | 289.649 | 13.225x | 11–0 |
| 524,288 | 128 | 11 | 2,104.544 | 196.406 | 10.715x | 11–0 |
| 524,288 | 256 | 11 | 1,258.367 | 142.906 | 8.806x | 11–0 |
| 1,048,576 | 64 | 11 | 4,239.782 | 269.872 | 15.710x | 11–0 |
| 1,048,576 | 128 | 11 | 2,271.497 | 201.285 | 11.285x | 11–0 |
| 1,048,576 | 256 | 11 | 1,362.724 | 138.571 | 9.834x | 11–0 |
| 2,097,152 | 64 | 11 | 3,921.761 | 352.779 | 11.117x | 11–0 |
| 2,097,152 | 128 | 11 | 2,173.928 | 212.629 | 10.224x | 11–0 |
| 2,097,152 | 256 | 11 | 1,284.964 | 145.651 | 8.822x | 11–0 |
| 4,194,304 | 64 | 11 | 4,063.277 | 468.227 | 8.678x | 11–0 |
| 4,194,304 | 128 | 11 | 2,188.840 | 260.597 | 8.399x | 11–0 |
| 4,194,304 | 256 | 11 | 1,310.588 | 154.984 | 8.456x | 11–0 |
| 8,388,607 | 64 | 11 | 4,083.423 | 429.227 | 9.513x | 11–0 |
| 8,388,607 | 128 | 11 | 2,231.870 | 258.497 | 8.634x | 11–0 |
| 8,388,607 | 256 | 11 | 1,323.979 | 158.245 | 8.367x | 11–0 |

#### Count

[PNG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-count-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 7 | 18.926 | 2.628 | 7.201x | 7–0 |
| 1,024 | 128 | 7 | 10.637 | 1.417 | 7.509x | 7–0 |
| 1,024 | 256 | 7 | 5.761 | 0.736 | 7.832x | 7–0 |
| 2,048 | 64 | 7 | 36.015 | 5.219 | 6.901x | 7–0 |
| 2,048 | 128 | 7 | 20.735 | 2.795 | 7.419x | 7–0 |
| 2,048 | 256 | 7 | 10.843 | 1.397 | 7.759x | 7–0 |
| 4,096 | 64 | 7 | 71.442 | 10.275 | 6.953x | 7–0 |
| 4,096 | 128 | 7 | 39.748 | 5.482 | 7.250x | 7–0 |
| 4,096 | 256 | 7 | 21.247 | 2.783 | 7.635x | 7–0 |
| 8,192 | 64 | 7 | 135.711 | 19.854 | 6.835x | 7–0 |
| 8,192 | 128 | 7 | 77.549 | 10.811 | 7.173x | 7–0 |
| 8,192 | 256 | 7 | 41.806 | 5.502 | 7.598x | 7–0 |
| 16,384 | 64 | 7 | 265.363 | 37.619 | 7.054x | 7–0 |
| 16,384 | 128 | 7 | 146.801 | 20.326 | 7.222x | 7–0 |
| 16,384 | 256 | 7 | 80.636 | 10.891 | 7.404x | 7–0 |
| 32,768 | 64 | 7 | 479.710 | 59.630 | 8.045x | 7–0 |
| 32,768 | 128 | 7 | 273.015 | 35.817 | 7.623x | 7–0 |
| 32,768 | 256 | 7 | 144.660 | 18.599 | 7.778x | 7–0 |
| 65,536 | 64 | 7 | 645.463 | 86.903 | 7.427x | 7–0 |
| 65,536 | 128 | 7 | 341.235 | 49.807 | 6.851x | 7–0 |
| 65,536 | 256 | 7 | 174.827 | 25.696 | 6.804x | 7–0 |
| 131,072 | 64 | 7 | 564.459 | 101.727 | 5.549x | 7–0 |
| 131,072 | 128 | 7 | 333.204 | 57.283 | 5.817x | 7–0 |
| 131,072 | 256 | 7 | 145.831 | 28.653 | 5.090x | 7–0 |
| 262,144 | 64 | 7 | 856.433 | 105.479 | 8.119x | 7–0 |
| 262,144 | 128 | 7 | 416.794 | 59.714 | 6.980x | 7–0 |
| 262,144 | 256 | 7 | 187.916 | 30.334 | 6.195x | 7–0 |
| 524,288 | 64 | 7 | 857.742 | 117.808 | 7.281x | 7–0 |
| 524,288 | 128 | 7 | 473.289 | 65.624 | 7.212x | 7–0 |
| 524,288 | 256 | 7 | 195.920 | 33.041 | 5.930x | 7–0 |
| 1,048,576 | 64 | 7 | 815.896 | 125.057 | 6.524x | 7–0 |
| 1,048,576 | 128 | 7 | 460.719 | 68.853 | 6.691x | 7–0 |
| 1,048,576 | 256 | 7 | 209.938 | 34.440 | 6.096x | 7–0 |
| 2,097,152 | 64 | 7 | 916.192 | 136.311 | 6.721x | 7–0 |
| 2,097,152 | 128 | 7 | 530.190 | 72.567 | 7.306x | 7–0 |
| 2,097,152 | 256 | 7 | 239.129 | 35.977 | 6.647x | 7–0 |
| 4,194,304 | 64 | 7 | 928.941 | 149.644 | 6.208x | 7–0 |
| 4,194,304 | 128 | 7 | 570.130 | 77.203 | 7.385x | 7–0 |
| 4,194,304 | 256 | 7 | 284.332 | 37.274 | 7.628x | 7–0 |
| 8,388,607 | 64 | 7 | 1,055.349 | 157.050 | 6.720x | 7–0 |
| 8,388,607 | 128 | 7 | 581.410 | 78.888 | 7.370x | 7–0 |
| 8,388,607 | 256 | 7 | 285.947 | 37.429 | 7.640x | 7–0 |

#### Extract

[PNG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-extract-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 3 | 8.509 | 2.924 | 2.910x | 3–0 |
| 1,024 | 128 | 3 | 8.196 | 2.668 | 3.072x | 3–0 |
| 1,024 | 256 | 3 | 8.480 | 2.349 | 3.610x | 3–0 |
| 2,048 | 64 | 3 | 16.241 | 5.897 | 2.754x | 3–0 |
| 2,048 | 128 | 3 | 16.699 | 5.264 | 3.172x | 3–0 |
| 2,048 | 256 | 3 | 16.504 | 4.738 | 3.483x | 3–0 |
| 4,096 | 64 | 3 | 31.943 | 11.598 | 2.754x | 3–0 |
| 4,096 | 128 | 3 | 34.090 | 10.269 | 3.320x | 3–0 |
| 4,096 | 256 | 3 | 27.925 | 8.900 | 3.138x | 3–0 |
| 8,192 | 64 | 3 | 64.958 | 21.289 | 3.051x | 3–0 |
| 8,192 | 128 | 3 | 51.891 | 17.949 | 2.891x | 3–0 |
| 8,192 | 256 | 3 | 55.135 | 17.628 | 3.128x | 3–0 |
| 16,384 | 64 | 3 | 108.647 | 33.318 | 3.261x | 3–0 |
| 16,384 | 128 | 3 | 82.507 | 33.403 | 2.470x | 3–0 |
| 16,384 | 256 | 3 | 75.941 | 30.166 | 2.517x | 3–0 |
| 32,768 | 64 | 3 | 131.552 | 53.188 | 2.473x | 3–0 |
| 32,768 | 128 | 3 | 130.773 | 47.978 | 2.726x | 3–0 |
| 32,768 | 256 | 3 | 131.115 | 42.888 | 3.057x | 3–0 |
| 65,536 | 64 | 3 | 145.834 | 55.335 | 2.635x | 3–0 |
| 65,536 | 128 | 3 | 139.335 | 47.472 | 2.935x | 3–0 |
| 65,536 | 256 | 3 | 137.962 | 41.612 | 3.315x | 3–0 |
| 131,072 | 64 | 3 | 173.261 | 63.284 | 2.738x | 3–0 |
| 131,072 | 128 | 3 | 167.207 | 54.149 | 3.088x | 3–0 |
| 131,072 | 256 | 3 | 165.569 | 47.854 | 3.460x | 3–0 |
| 262,144 | 64 | 3 | 202.466 | 66.165 | 3.060x | 3–0 |
| 262,144 | 128 | 3 | 197.743 | 56.797 | 3.482x | 3–0 |
| 262,144 | 256 | 3 | 193.332 | 50.801 | 3.806x | 3–0 |
| 524,288 | 64 | 3 | 203.553 | 73.239 | 2.779x | 3–0 |
| 524,288 | 128 | 3 | 201.673 | 62.277 | 3.238x | 3–0 |
| 524,288 | 256 | 3 | 198.797 | 55.554 | 3.578x | 3–0 |
| 1,048,576 | 64 | 3 | 230.633 | 77.557 | 2.974x | 3–0 |
| 1,048,576 | 128 | 3 | 226.930 | 65.767 | 3.451x | 3–0 |
| 1,048,576 | 256 | 3 | 225.608 | 58.447 | 3.860x | 3–0 |
| 2,097,152 | 64 | 3 | 264.030 | 109.156 | 2.419x | 3–0 |
| 2,097,152 | 128 | 3 | 263.559 | 85.244 | 3.092x | 3–0 |
| 2,097,152 | 256 | 3 | 260.285 | 73.784 | 3.528x | 3–0 |
| 4,194,304 | 64 | 3 | 391.582 | 139.425 | 2.809x | 3–0 |
| 4,194,304 | 128 | 3 | 400.770 | 106.479 | 3.764x | 3–0 |
| 4,194,304 | 256 | 3 | 380.299 | 87.314 | 4.356x | 3–0 |
| 8,388,607 | 64 | 3 | 502.313 | 115.609 | 4.345x | 3–0 |
| 8,388,607 | 128 | 3 | 384.466 | 102.622 | 3.746x | 3–0 |
| 8,388,607 | 256 | 3 | 490.066 | 86.902 | 5.639x | 3–0 |

#### Replace

[PNG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-replace-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 14 | 7.141 | 0.902 | 7.919x | 14–0 |
| 1,024 | 128 | 14 | 4.336 | 0.476 | 9.104x | 14–0 |
| 1,024 | 256 | 14 | 2.393 | 0.242 | 9.902x | 14–0 |
| 2,048 | 64 | 14 | 13.602 | 1.747 | 7.785x | 14–0 |
| 2,048 | 128 | 14 | 8.394 | 0.923 | 9.095x | 14–0 |
| 2,048 | 256 | 14 | 4.510 | 0.450 | 10.014x | 14–0 |
| 4,096 | 64 | 14 | 26.856 | 3.431 | 7.827x | 14–0 |
| 4,096 | 128 | 14 | 16.194 | 1.808 | 8.958x | 14–0 |
| 4,096 | 256 | 14 | 8.608 | 0.888 | 9.690x | 14–0 |
| 8,192 | 64 | 14 | 51.456 | 6.656 | 7.731x | 14–0 |
| 8,192 | 128 | 14 | 29.997 | 3.545 | 8.463x | 14–0 |
| 8,192 | 256 | 14 | 16.519 | 1.765 | 9.357x | 14–0 |
| 16,384 | 64 | 14 | 99.345 | 12.336 | 8.053x | 14–0 |
| 16,384 | 128 | 14 | 54.338 | 6.350 | 8.557x | 14–0 |
| 16,384 | 256 | 14 | 30.810 | 3.248 | 9.486x | 14–0 |
| 32,768 | 64 | 14 | 149.969 | 19.355 | 7.748x | 14–0 |
| 32,768 | 128 | 14 | 93.270 | 10.310 | 9.046x | 14–0 |
| 32,768 | 256 | 14 | 53.192 | 5.184 | 10.260x | 14–0 |
| 65,536 | 64 | 14 | 198.756 | 23.242 | 8.552x | 14–0 |
| 65,536 | 128 | 14 | 112.954 | 11.789 | 9.582x | 14–0 |
| 65,536 | 256 | 14 | 59.226 | 5.605 | 10.567x | 14–0 |
| 131,072 | 64 | 14 | 245.120 | 28.267 | 8.672x | 14–0 |
| 131,072 | 128 | 14 | 133.264 | 14.298 | 9.320x | 14–0 |
| 131,072 | 256 | 14 | 67.452 | 6.733 | 10.018x | 14–0 |
| 262,144 | 64 | 14 | 230.274 | 32.028 | 7.190x | 14–0 |
| 262,144 | 128 | 14 | 124.408 | 16.048 | 7.752x | 14–0 |
| 262,144 | 256 | 14 | 64.617 | 7.592 | 8.511x | 14–0 |
| 524,288 | 64 | 14 | 253.222 | 34.836 | 7.269x | 14–0 |
| 524,288 | 128 | 14 | 135.726 | 17.413 | 7.795x | 14–0 |
| 524,288 | 256 | 14 | 75.493 | 8.284 | 9.113x | 14–0 |
| 1,048,576 | 64 | 14 | 273.909 | 37.220 | 7.359x | 14–0 |
| 1,048,576 | 128 | 14 | 156.740 | 18.536 | 8.456x | 14–0 |
| 1,048,576 | 256 | 14 | 83.612 | 8.767 | 9.538x | 14–0 |
| 2,097,152 | 64 | 14 | 312.889 | 39.464 | 7.929x | 14–0 |
| 2,097,152 | 128 | 14 | 170.990 | 19.313 | 8.854x | 14–0 |
| 2,097,152 | 256 | 14 | 90.000 | 9.006 | 9.993x | 14–0 |
| 4,194,304 | 64 | 14 | 342.718 | 40.610 | 8.439x | 14–0 |
| 4,194,304 | 128 | 14 | 184.314 | 19.633 | 9.388x | 14–0 |
| 4,194,304 | 256 | 14 | 92.362 | 9.126 | 10.121x | 14–0 |
| 8,388,607 | 64 | 14 | 371.238 | 40.698 | 9.122x | 14–0 |
| 8,388,607 | 128 | 14 | 189.581 | 19.448 | 9.748x | 14–0 |
| 8,388,607 | 256 | 14 | 93.615 | 8.968 | 10.439x | 14–0 |

#### Split

[PNG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-split-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 7 | 4.543 | 1.098 | 4.137x | 7–0 |
| 1,024 | 128 | 7 | 3.257 | 0.620 | 5.252x | 7–0 |
| 1,024 | 256 | 7 | 2.123 | 0.334 | 6.361x | 7–0 |
| 2,048 | 64 | 7 | 8.843 | 2.137 | 4.138x | 7–0 |
| 2,048 | 128 | 7 | 6.125 | 1.218 | 5.029x | 7–0 |
| 2,048 | 256 | 7 | 3.959 | 0.623 | 6.356x | 7–0 |
| 4,096 | 64 | 7 | 15.921 | 4.257 | 3.740x | 7–0 |
| 4,096 | 128 | 7 | 11.678 | 2.330 | 5.013x | 7–0 |
| 4,096 | 256 | 7 | 7.296 | 1.205 | 6.056x | 7–0 |
| 8,192 | 64 | 7 | 29.491 | 7.873 | 3.746x | 7–0 |
| 8,192 | 128 | 7 | 21.013 | 4.466 | 4.705x | 7–0 |
| 8,192 | 256 | 7 | 13.371 | 2.367 | 5.648x | 7–0 |
| 16,384 | 64 | 7 | 51.612 | 13.849 | 3.727x | 7–0 |
| 16,384 | 128 | 7 | 32.626 | 7.786 | 4.190x | 7–0 |
| 16,384 | 256 | 7 | 21.056 | 4.379 | 4.809x | 7–0 |
| 32,768 | 64 | 7 | 72.257 | 20.495 | 3.526x | 7–0 |
| 32,768 | 128 | 7 | 47.772 | 12.566 | 3.802x | 7–0 |
| 32,768 | 256 | 7 | 29.618 | 6.712 | 4.413x | 7–0 |
| 65,536 | 64 | 7 | 89.775 | 26.546 | 3.382x | 7–0 |
| 65,536 | 128 | 7 | 54.028 | 15.171 | 3.561x | 7–0 |
| 65,536 | 256 | 7 | 32.795 | 8.326 | 3.939x | 7–0 |
| 131,072 | 64 | 7 | 101.674 | 31.473 | 3.231x | 7–0 |
| 131,072 | 128 | 7 | 65.823 | 18.540 | 3.550x | 7–0 |
| 131,072 | 256 | 7 | 39.947 | 9.796 | 4.078x | 7–0 |
| 262,144 | 64 | 7 | 103.051 | 34.741 | 2.966x | 7–0 |
| 262,144 | 128 | 7 | 71.647 | 20.544 | 3.488x | 7–0 |
| 262,144 | 256 | 7 | 43.244 | 10.790 | 4.008x | 7–0 |
| 524,288 | 64 | 7 | 121.706 | 40.310 | 3.019x | 7–0 |
| 524,288 | 128 | 7 | 82.533 | 23.005 | 3.588x | 7–0 |
| 524,288 | 256 | 7 | 51.628 | 12.095 | 4.269x | 7–0 |
| 1,048,576 | 64 | 7 | 149.656 | 44.840 | 3.338x | 7–0 |
| 1,048,576 | 128 | 7 | 104.523 | 25.120 | 4.161x | 7–0 |
| 1,048,576 | 256 | 7 | 65.478 | 12.889 | 5.080x | 7–0 |
| 2,097,152 | 64 | 7 | 195.371 | 49.663 | 3.934x | 7–0 |
| 2,097,152 | 128 | 7 | 121.755 | 27.167 | 4.482x | 7–0 |
| 2,097,152 | 256 | 7 | 74.933 | 13.707 | 5.467x | 7–0 |
| 4,194,304 | 64 | 7 | 237.132 | 55.373 | 4.282x | 7–0 |
| 4,194,304 | 128 | 7 | 138.629 | 29.209 | 4.746x | 7–0 |
| 4,194,304 | 256 | 7 | 82.452 | 14.256 | 5.784x | 7–0 |
| 8,388,607 | 64 | 7 | 286.880 | 58.857 | 4.874x | 7–0 |
| 8,388,607 | 128 | 7 | 169.390 | 29.920 | 5.661x | 7–0 |
| 8,388,607 | 256 | 7 | 57.509 | 14.237 | 4.039x | 7–0 |

[All API measurements](docs/_static/benchmarks/cudf-api-throughput-data.csv)

<!-- END GENERATED CUDF API RESULTS -->

The `regex_ir/cold` and `cudf/cold` registrations remain available for
one-shot setup-plus-first-execution measurements. Their separate `JIT Ready`
summary stops after module loading and function lookup, before output
allocation or the first launch. Production integrations should cache the
linked cubin by pattern, options, architecture, and toolkit version.

At one million 128-byte rows, the current warm and cold measurements are:

| Pattern | Regex IR warm (ms) | cuDF warm (ms) | Warm speedup | Regex IR cold JIT + first launch (ms) | cuDF program + first call (ms) |
|:---|---:|---:|---:|---:|---:|
| log | 0.654 | 2.089 | 3.194x | 10.704 | 2.088 |
| email | 0.666 | 9.562 | 14.361x | 10.826 | 9.646 |
| URI | 0.663 | 10.887 | 16.429x | 11.122 | 11.012 |
| IPv4 | 0.676 | 28.423 | 42.056x | 11.415 | 28.527 |

The log, email, and URI cold paths do not repay roughly 10 ms of device
compilation on the first call; URI is approximately break-even. IPv4 already
repays that cost on one million rows. Cached use avoids it entirely.

### Suites B–E results: imported complete corpora

The external suites were run on the same RTX A6000 software stack, with at
least five samples, 0.05 seconds minimum measurement time, 2% target noise, and
a 20-second timeout. OpenResty, Leipzig, Boost long-Twain, mariomka, and Boost's
repeated scalar cases used 32,768 rows in each of eight columns and a 256-byte
maximum. Boost's 50 KiB medium-Twain, 34 KiB C++, and 52 KiB HTML inputs used
1,024 rows in one column with a 64-byte maximum so the measurement was not
dominated by hundreds of thousands of empty strings. All 176 engine states
completed, and every pre-timing Regex IR/cuDF output comparison passed.

Latency is NVBench's mean GPU time for applying one compiled expression to all
eight columns. The arithmetic means expose absolute long-tail cost; geometric
speedup gives every expression equal weight.

| Source suite | Cases | Regex IR mean (ms) | cuDF mean (ms) | Geometric speedup | Per-case wins |
|:---|---:|---:|---:|---:|:---:|
| OpenResty | 31 | 0.354 | 9.673 | 19.445x | 31–0 |
| Rust Leipzig | 18 | 0.431 | 9.659 | 20.866x | 18–0 |
| Boost/GCC | 36 | 0.178 | 5.321 | 19.705x | 36–0 |
| mariomka | 3 | 0.260 | 6.865 | 24.450x | 3–0 |
| **All imported suites** | **88** | **0.295** | **7.794** | **19.991x** | **88–0** |

`JIT Ready` is recorded once during each state setup and is not included in the
execution latency above:

| Source suite | Regex IR JIT-ready mean (ms) | cuDF program-create mean (ms) | Regex IR JIT-ready median (ms) | cuDF median (ms) |
|:---|---:|---:|---:|---:|
| OpenResty | 12.447 | 0.0162 | 10.867 | 0.0076 |
| Rust Leipzig | 17.468 | 0.0232 | 10.934 | 0.0094 |
| Boost/GCC | 17.723 | 0.0391 | 11.582 | 0.0108 |
| mariomka | 10.839 | 0.0117 | 10.840 | 0.0112 |
| **All imported suites** | **15.578** | **0.0269** | **11.066** | **0.0091** |

These setup columns intentionally measure different work. Regex IR's value is
a true device-code path—parse/lower, render NVVM IR, libNVVM `-opt=3 -gen-lto`,
cache-disabled nvJitLink device LTO at `-O3`, module load, and function lookup.
cuDF creates an interpreter program and does not JIT a specialized CUDA
kernel.

The subgroup table makes the workload intent visible instead of blending very
different corpora into one score:

| Suite | Subgroup and intention | Cases | Regex IR mean (ms) | cuDF mean (ms) | Geometric speedup | Wins |
|:---|:---|---:|---:|---:|---:|:---:|
| OpenResty | full-size alphabet/delimiter engine stress | 16 | 0.291 | 8.355 | 18.204x | 16–0 |
| OpenResty | complete `mtent12` search | 15 | 0.422 | 11.078 | 20.862x | 15–0 |
| Rust Leipzig | complete `3200.txt` text search | 15 | 0.461 | 10.171 | 21.210x | 15–0 |
| Rust Leipzig | Unicode literals/property on `3200.txt` | 2 | 0.273 | 3.327 | 12.083x | 2–0 |
| Rust Leipzig | CSV stress expression on `3200.txt` | 1 | 0.301 | 14.639 | 48.694x | 1–0 |
| Boost/GCC | complete long-Twain search | 6 | 0.793 | 9.296 | 11.217x | 6–0 |
| Boost/GCC | exact 50 KiB medium-Twain search | 6 | 0.055 | 0.292 | 5.329x | 6–0 |
| Boost/GCC | complete 34 KiB `crc.hpp` scan | 4 | 0.056 | 20.655 | 23.813x | 4–0 |
| Boost/GCC | complete 52 KiB `libraries.htm` scan | 6 | 0.038 | 0.390 | 9.516x | 6–0 |
| Boost/GCC | exact repeated scalar records | 14 | 0.063 | 3.505 | 56.864x | 14–0 |
| mariomka | complete mixed-language input | 3 | 0.260 | 6.865 | 24.450x | 3–0 |

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

These cases were rerun on 2026-07-05 with at least five samples, 0.05 seconds of
measured GPU time, a 2% target-noise threshold, and a 10-second per-state timeout.
All 148 engine states completed without warnings, skips, or timeouts, and every
pre-timing Regex IR/cuDF output comparison passed.

| Source suite | Corpus expressions | Regex IR geometric throughput (GiB/s) | cuDF geometric throughput (GiB/s) | Geometric speedup | Pair wins | Regex IR cold JIT mean (ms) | cuDF program-create mean (ms) |
|:---|---:|---:|---:|---:|:---:|---:|---:|
| OpenResty | 31 | 54.659 | 2.716 | 20.125x | 31–0 | 14.062 | 0.0162 |
| Rust Leipzig | 18 | 41.622 | 1.947 | 21.376x | 18–0 | 17.739 | 0.0235 |
| Boost/GCC | 22 | 3.417 | 0.266 | 12.830x | 22–0 | 23.927 | 0.0560 |
| mariomka | 3 | 24.690 | 0.987 | 25.024x | 3–0 | 11.048 | 0.0124 |
| **All complete-corpus suites** | **74** | **21.724** | **1.205** | **18.022x** | **74–0** | **17.767** | **0.0297** |

Regex IR won 74 of 74 paired full-corpus cases. The narrowest result was OpenResty case
1 (3.790x); the largest was Boost/GCC case 14 (679.602x).

#### OpenResty full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-openresty-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | literal miss | 25.000 | 0.282 | 1.071 | 3.790x |
| 2 | short alt miss | 10.000 | 0.128 | 2.529 | 19.810x |
| 3 | suffix alt miss | 25.000 | 0.384 | 8.508 | 22.137x |
| 4 | suffix alt prose | 19.117 | 0.420 | 9.063 | 21.602x |
| 5 | wide class miss | 25.000 | 0.331 | 3.629 | 10.957x |
| 6 | split class miss | 25.000 | 0.337 | 3.691 | 10.937x |
| 7 | split class prose | 19.117 | 0.187 | 2.328 | 12.446x |
| 8 | large alt prose | 19.117 | 0.577 | 32.788 | 56.819x |
| 9 | large alt miss | 25.000 | 0.377 | 31.798 | 84.309x |
| 10 | nested alt | 25.000 | 0.370 | 8.078 | 21.838x |
| 11 | long nested alt miss | 10.000 | 0.170 | 8.598 | 50.720x |
| 12 | capture chain miss | 25.000 | 0.370 | 8.927 | 24.159x |
| 13 | capture chain random miss | 10.000 | 0.189 | 8.558 | 45.386x |
| 14 | lazy class repeat | 10.000 | 0.120 | 1.352 | 11.250x |
| 15 | lazy dot repeat | 10.000 | 0.119 | 1.344 | 11.275x |
| 16 | greedy dot repeat | 10.000 | 0.121 | 1.348 | 11.182x |
| 17 | anchored literal | 19.117 | 0.406 | 3.921 | 9.665x |
| 18 | literal prose | 19.117 | 0.260 | 1.755 | 6.739x |
| 19 | folded literal | 19.117 | 0.392 | 4.469 | 11.413x |
| 20 | class suffix | 19.117 | 0.418 | 5.935 | 14.207x |
| 21 | name alternation | 19.117 | 0.401 | 5.552 | 13.845x |
| 22 | word boundary | 19.117 | 0.456 | 9.255 | 20.316x |
| 23 | negated bounded | 19.117 | 0.499 | 14.870 | 29.800x |
| 24 | name literals | 19.117 | 0.439 | 8.833 | 20.129x |
| 25 | folded names | 19.117 | 0.502 | 10.751 | 21.419x |
| 26 | short prefix names | 19.117 | 0.446 | 27.909 | 62.522x |
| 27 | required prefix names | 19.117 | 0.446 | 28.745 | 64.412x |
| 28 | word suffix | 19.117 | 0.380 | 7.711 | 20.317x |
| 29 | bounded word suffix | 19.117 | 0.485 | 12.227 | 25.229x |
| 30 | captured name suffix | 19.117 | 0.448 | 9.507 | 21.230x |
| 31 | quoted sentence | 19.117 | 0.406 | 14.296 | 35.233x |

#### Rust Leipzig full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-leipzig-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | twain | 15.272 | 0.207 | 1.607 | 7.760x |
| 2 | twain ignore case | 15.272 | 0.316 | 3.716 | 11.743x |
| 3 | shing | 15.272 | 0.339 | 4.949 | 14.592x |
| 4 | huck saw | 15.272 | 0.327 | 4.927 | 15.070x |
| 5 | word nn | 15.272 | 0.362 | 7.504 | 20.755x |
| 6 | negated bounded | 15.272 | 0.395 | 12.541 | 31.711x |
| 7 | names | 15.272 | 0.357 | 7.778 | 21.791x |
| 8 | names ignore case | 15.272 | 0.409 | 9.228 | 22.570x |
| 9 | optional prefix | 15.272 | 0.356 | 24.224 | 67.960x |
| 10 | required prefix | 15.272 | 0.363 | 24.033 | 66.117x |
| 11 | tom river | 15.272 | 1.412 | 13.380 | 9.474x |
| 12 | word ing | 15.272 | 0.306 | 6.339 | 20.682x |
| 13 | bounded ing | 15.272 | 0.395 | 10.092 | 25.551x |
| 14 | name suffix | 15.272 | 0.366 | 7.818 | 21.343x |
| 15 | quoted sentence | 15.272 | 0.331 | 13.663 | 41.325x |
| 16 | unicode symbols | 15.272 | 0.274 | 3.693 | 13.457x |
| 17 | math symbol property | 15.272 | 0.279 | 2.913 | 10.435x |
| 18 | csv field | 15.272 | 0.305 | 14.634 | 48.048x |

#### Boost/GCC full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-boost-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | long twain | 19.117 | 0.263 | 1.773 | 6.749x |
| 2 | long huck | 19.117 | 0.258 | 2.041 | 7.914x |
| 3 | long ing | 19.117 | 0.382 | 7.786 | 20.407x |
| 4 | long line twain | 19.117 | 0.411 | 7.357 | 17.920x |
| 5 | long names | 19.117 | 0.442 | 9.063 | 20.510x |
| 6 | long names near river | 19.117 | 2.065 | 27.632 | 13.383x |
| 7 | medium twain | 0.048 | 0.021 | 0.087 | 4.203x |
| 8 | medium huck | 0.048 | 0.019 | 0.078 | 4.089x |
| 9 | medium ing | 0.048 | 0.025 | 0.344 | 13.853x |
| 10 | medium line twain | 0.048 | 0.027 | 0.343 | 12.648x |
| 11 | medium names | 0.048 | 0.028 | 0.285 | 10.246x |
| 12 | medium names near river | 0.048 | 0.125 | 0.635 | 5.086x |
| 13 | cpp declaration | 0.033 | 0.027 | 0.340 | 12.499x |
| 14 | cpp tokens | 0.033 | 0.121 | 82.473 | 679.602x |
| 15 | cpp include | 0.033 | 0.026 | 0.227 | 8.656x |
| 16 | boost include | 0.033 | 0.028 | 0.234 | 8.226x |
| 17 | html names | 0.049 | 0.032 | 0.285 | 8.853x |
| 18 | html paragraph | 0.049 | 0.025 | 0.287 | 11.509x |
| 19 | html anchor | 0.049 | 0.029 | 0.774 | 26.531x |
| 20 | html heading | 0.049 | 0.025 | 0.304 | 12.025x |
| 21 | html image | 0.049 | 0.027 | 0.426 | 16.008x |
| 22 | html font | 0.049 | 0.030 | 0.275 | 9.074x |

#### mariomka full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-mariomka-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | email | 6.523 | 0.252 | 3.875 | 15.359x |
| 2 | uri | 6.523 | 0.266 | 6.935 | 26.102x |
| 3 | ipv4 | 6.523 | 0.256 | 10.014 | 39.087x |

[All complete-corpus measurements](docs/_static/benchmarks/corpus-throughput-data.csv)

<!-- END GENERATED CORPUS SWEEP RESULTS -->

The previous assertion outliers are gone. Boolean word boundaries, line
anchors, end assertions, and CRLF-sensitive assertions now use an
assertion-aware deterministic table instead of falling back to recursive
Thompson execution. The full-corpus word-boundary and FTP expressions are
included in the 88–0 result rather than excluded or hidden.

The imported workloads also drove two code-generation changes. Large boolean
alternations are split into independently optimized DFA functions and joined
by a short-circuit wrapper: the Leipzig bounded `Tom...river` case fell from
about 50 seconds to 6.10 ms, versus cuDF at 31.13 ms. Boolean machines reclaim
the unused priority flag as a fifteenth state bit; the 16,385-state bounded
negated-class case now runs in about 0.72 ms instead of 94–109 ms, versus cuDF
at about 25.8 ms. Transition tables that exceed the 64 KiB constant segment are
emitted into read-only global storage.

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
policy. The benchmark wrappers use 256-thread blocks, selected from the
register/occupancy profile, and write extract captures directly to final device
storage.

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
`.nvvm.ll`, libNVVM `.ptx`, and linked `.cubin` files for `cuobjdump` or
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
