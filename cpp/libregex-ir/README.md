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
libfmt's `fmt::format`. A consumer passes that string to libNVVM, links the
resulting PTX to its kernel with nvJitLink, then launches the kernel through its
existing CUDA runtime.

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
replacement, and regex-split APIs with Regex IR's NVVM IR → libNVVM PTX →
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
| warm/cold JIT | `regex_ir/{warm,cold}` and `cudf/{warm,cold}` | Four log/email/URI/IPv4 expressions over fixed-width rows | Cached execution versus compilation, linking, module loading, and first launch |
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
outside the execution-latency sample. Their uncached setup duration is exposed
separately as `Cold Compile`: Regex IR includes parsing/lowering, NVVM IR
rendering, libNVVM, nvJitLink, and module load; cuDF includes
`regex_program::create`. Both engines consume the same owning cuDF STRING
columns and allocate owning cuDF result objects inside the timed call.

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
| contains | 11 | 42 | 462 | 632.635 | 78.318 | 8.078x | 462–0 |
| count | 7 | 42 | 294 | 156.353 | 25.174 | 6.211x | 294–0 |
| extract | 3 | 42 | 126 | 99.794 | 33.819 | 2.951x | 126–0 |
| replace | 14 | 42 | 588 | 51.202 | 7.163 | 7.149x | 588–0 |
| split | 7 | 42 | 294 | 35.583 | 9.413 | 3.780x | 294–0 |
| **All APIs** | **42** | **42** | **1764** | **117.603** | **19.321** | **6.087x** | **1764–0** |

Regex IR won 1764 of 1764 paired measurements. Contains, count, extract, replace, and
split won every state. The narrowest result was split pattern 1 at 8,192 rows/64 bytes
(1.095x), while the largest was contains pattern 2 at 8,388,607 rows/64 bytes (51.790x).
The results describe this hardware, corpus, and allocation contract—not every regex or
deployment. Use the commands above to reproduce the matrix.

The tables below report geometric mean throughput across each API's expression
cases independently at every row-count/`StringBytes` geometry.

#### Contains

[PNG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-contains-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-contains-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 11 | 38.450 | 9.020 | 4.263x | 11–0 |
| 1,024 | 128 | 11 | 27.150 | 5.946 | 4.566x | 11–0 |
| 1,024 | 256 | 11 | 18.984 | 3.775 | 5.028x | 11–0 |
| 2,048 | 64 | 11 | 77.266 | 17.996 | 4.293x | 11–0 |
| 2,048 | 128 | 11 | 53.933 | 11.836 | 4.557x | 11–0 |
| 2,048 | 256 | 11 | 37.857 | 7.525 | 5.031x | 11–0 |
| 4,096 | 64 | 11 | 153.687 | 35.988 | 4.271x | 11–0 |
| 4,096 | 128 | 11 | 107.022 | 23.604 | 4.534x | 11–0 |
| 4,096 | 256 | 11 | 75.211 | 14.856 | 5.063x | 11–0 |
| 8,192 | 64 | 11 | 305.790 | 60.867 | 5.024x | 11–0 |
| 8,192 | 128 | 11 | 209.936 | 35.825 | 5.860x | 11–0 |
| 8,192 | 256 | 11 | 148.164 | 23.502 | 6.304x | 11–0 |
| 16,384 | 64 | 11 | 605.499 | 84.479 | 7.167x | 11–0 |
| 16,384 | 128 | 11 | 412.499 | 61.694 | 6.686x | 11–0 |
| 16,384 | 256 | 11 | 291.210 | 43.548 | 6.687x | 11–0 |
| 32,768 | 64 | 11 | 1,157.290 | 119.602 | 9.676x | 11–0 |
| 32,768 | 128 | 11 | 774.509 | 90.676 | 8.542x | 11–0 |
| 32,768 | 256 | 11 | 532.944 | 63.239 | 8.427x | 11–0 |
| 65,536 | 64 | 11 | 1,425.528 | 145.906 | 9.770x | 11–0 |
| 65,536 | 128 | 11 | 879.453 | 110.018 | 7.994x | 11–0 |
| 65,536 | 256 | 11 | 577.769 | 80.848 | 7.146x | 11–0 |
| 131,072 | 64 | 11 | 2,198.330 | 160.286 | 13.715x | 11–0 |
| 131,072 | 128 | 11 | 1,327.820 | 116.458 | 11.402x | 11–0 |
| 131,072 | 256 | 11 | 842.588 | 85.295 | 9.879x | 11–0 |
| 262,144 | 64 | 11 | 3,026.495 | 178.596 | 16.946x | 11–0 |
| 262,144 | 128 | 11 | 1,772.973 | 129.892 | 13.650x | 11–0 |
| 262,144 | 256 | 11 | 1,087.557 | 94.987 | 11.450x | 11–0 |
| 524,288 | 64 | 11 | 3,758.721 | 289.672 | 12.976x | 11–0 |
| 524,288 | 128 | 11 | 2,117.107 | 196.405 | 10.779x | 11–0 |
| 524,288 | 256 | 11 | 1,276.596 | 143.101 | 8.921x | 11–0 |
| 1,048,576 | 64 | 11 | 4,299.463 | 269.869 | 15.932x | 11–0 |
| 1,048,576 | 128 | 11 | 2,352.264 | 201.204 | 11.691x | 11–0 |
| 1,048,576 | 256 | 11 | 1,398.421 | 138.589 | 10.090x | 11–0 |
| 2,097,152 | 64 | 11 | 4,011.468 | 352.701 | 11.374x | 11–0 |
| 2,097,152 | 128 | 11 | 2,244.011 | 212.978 | 10.536x | 11–0 |
| 2,097,152 | 256 | 11 | 1,351.430 | 145.639 | 9.279x | 11–0 |
| 4,194,304 | 64 | 11 | 4,193.493 | 468.183 | 8.957x | 11–0 |
| 4,194,304 | 128 | 11 | 2,323.461 | 260.757 | 8.910x | 11–0 |
| 4,194,304 | 256 | 11 | 1,388.328 | 155.255 | 8.942x | 11–0 |
| 8,388,607 | 64 | 11 | 4,277.635 | 429.251 | 9.965x | 11–0 |
| 8,388,607 | 128 | 11 | 2,357.594 | 258.505 | 9.120x | 11–0 |
| 8,388,607 | 256 | 11 | 1,404.772 | 166.193 | 8.453x | 11–0 |

#### Count

[PNG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-count-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-count-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 7 | 14.711 | 2.634 | 5.584x | 7–0 |
| 1,024 | 128 | 7 | 8.251 | 1.422 | 5.804x | 7–0 |
| 1,024 | 256 | 7 | 4.599 | 0.739 | 6.227x | 7–0 |
| 2,048 | 64 | 7 | 28.328 | 5.235 | 5.411x | 7–0 |
| 2,048 | 128 | 7 | 15.835 | 2.802 | 5.651x | 7–0 |
| 2,048 | 256 | 7 | 8.660 | 1.402 | 6.177x | 7–0 |
| 4,096 | 64 | 7 | 55.893 | 10.332 | 5.410x | 7–0 |
| 4,096 | 128 | 7 | 30.665 | 5.484 | 5.592x | 7–0 |
| 4,096 | 256 | 7 | 16.950 | 2.789 | 6.077x | 7–0 |
| 8,192 | 64 | 7 | 106.646 | 19.888 | 5.362x | 7–0 |
| 8,192 | 128 | 7 | 59.587 | 10.851 | 5.492x | 7–0 |
| 8,192 | 256 | 7 | 33.675 | 5.520 | 6.100x | 7–0 |
| 16,384 | 64 | 7 | 209.328 | 37.785 | 5.540x | 7–0 |
| 16,384 | 128 | 7 | 114.114 | 20.404 | 5.593x | 7–0 |
| 16,384 | 256 | 7 | 64.744 | 10.913 | 5.933x | 7–0 |
| 32,768 | 64 | 7 | 369.404 | 59.656 | 6.192x | 7–0 |
| 32,768 | 128 | 7 | 221.485 | 35.838 | 6.180x | 7–0 |
| 32,768 | 256 | 7 | 119.881 | 18.643 | 6.430x | 7–0 |
| 65,536 | 64 | 7 | 487.475 | 86.688 | 5.623x | 7–0 |
| 65,536 | 128 | 7 | 275.358 | 49.803 | 5.529x | 7–0 |
| 65,536 | 256 | 7 | 144.243 | 25.725 | 5.607x | 7–0 |
| 131,072 | 64 | 7 | 569.853 | 101.902 | 5.592x | 7–0 |
| 131,072 | 128 | 7 | 330.715 | 57.387 | 5.763x | 7–0 |
| 131,072 | 256 | 7 | 144.742 | 28.702 | 5.043x | 7–0 |
| 262,144 | 64 | 7 | 774.012 | 105.639 | 7.327x | 7–0 |
| 262,144 | 128 | 7 | 433.901 | 59.725 | 7.265x | 7–0 |
| 262,144 | 256 | 7 | 185.192 | 30.352 | 6.102x | 7–0 |
| 524,288 | 64 | 7 | 855.947 | 117.983 | 7.255x | 7–0 |
| 524,288 | 128 | 7 | 476.282 | 65.680 | 7.252x | 7–0 |
| 524,288 | 256 | 7 | 213.623 | 33.091 | 6.456x | 7–0 |
| 1,048,576 | 64 | 7 | 917.957 | 124.934 | 7.348x | 7–0 |
| 1,048,576 | 128 | 7 | 509.731 | 68.937 | 7.394x | 7–0 |
| 1,048,576 | 256 | 7 | 235.041 | 34.488 | 6.815x | 7–0 |
| 2,097,152 | 64 | 7 | 948.109 | 136.297 | 6.956x | 7–0 |
| 2,097,152 | 128 | 7 | 528.666 | 72.619 | 7.280x | 7–0 |
| 2,097,152 | 256 | 7 | 247.527 | 35.842 | 6.906x | 7–0 |
| 4,194,304 | 64 | 7 | 956.794 | 149.264 | 6.410x | 7–0 |
| 4,194,304 | 128 | 7 | 531.880 | 77.099 | 6.899x | 7–0 |
| 4,194,304 | 256 | 7 | 255.771 | 37.362 | 6.846x | 7–0 |
| 8,388,607 | 64 | 7 | 971.252 | 157.120 | 6.182x | 7–0 |
| 8,388,607 | 128 | 7 | 540.555 | 78.839 | 6.856x | 7–0 |
| 8,388,607 | 256 | 7 | 260.414 | 37.463 | 6.951x | 7–0 |

#### Extract

[PNG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-extract-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-extract-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 3 | 6.730 | 2.924 | 2.301x | 3–0 |
| 1,024 | 128 | 3 | 6.777 | 2.668 | 2.540x | 3–0 |
| 1,024 | 256 | 3 | 6.739 | 2.349 | 2.869x | 3–0 |
| 2,048 | 64 | 3 | 13.487 | 5.897 | 2.287x | 3–0 |
| 2,048 | 128 | 3 | 13.414 | 5.264 | 2.548x | 3–0 |
| 2,048 | 256 | 3 | 13.329 | 4.738 | 2.813x | 3–0 |
| 4,096 | 64 | 3 | 24.941 | 11.598 | 2.150x | 3–0 |
| 4,096 | 128 | 3 | 27.105 | 10.269 | 2.639x | 3–0 |
| 4,096 | 256 | 3 | 22.794 | 8.900 | 2.561x | 3–0 |
| 8,192 | 64 | 3 | 50.105 | 21.289 | 2.354x | 3–0 |
| 8,192 | 128 | 3 | 42.912 | 17.949 | 2.391x | 3–0 |
| 8,192 | 256 | 3 | 45.244 | 17.628 | 2.567x | 3–0 |
| 16,384 | 64 | 3 | 87.812 | 33.318 | 2.636x | 3–0 |
| 16,384 | 128 | 3 | 67.775 | 33.403 | 2.029x | 3–0 |
| 16,384 | 256 | 3 | 68.867 | 30.166 | 2.283x | 3–0 |
| 32,768 | 64 | 3 | 110.686 | 53.188 | 2.081x | 3–0 |
| 32,768 | 128 | 3 | 115.953 | 47.978 | 2.417x | 3–0 |
| 32,768 | 256 | 3 | 115.346 | 42.888 | 2.689x | 3–0 |
| 65,536 | 64 | 3 | 136.097 | 55.335 | 2.459x | 3–0 |
| 65,536 | 128 | 3 | 131.443 | 47.472 | 2.769x | 3–0 |
| 65,536 | 256 | 3 | 129.542 | 41.612 | 3.113x | 3–0 |
| 131,072 | 64 | 3 | 165.306 | 63.284 | 2.612x | 3–0 |
| 131,072 | 128 | 3 | 163.227 | 54.149 | 3.014x | 3–0 |
| 131,072 | 256 | 3 | 162.173 | 47.854 | 3.389x | 3–0 |
| 262,144 | 64 | 3 | 197.443 | 66.165 | 2.984x | 3–0 |
| 262,144 | 128 | 3 | 192.854 | 56.797 | 3.396x | 3–0 |
| 262,144 | 256 | 3 | 189.370 | 50.801 | 3.728x | 3–0 |
| 524,288 | 64 | 3 | 203.923 | 73.239 | 2.784x | 3–0 |
| 524,288 | 128 | 3 | 200.771 | 62.277 | 3.224x | 3–0 |
| 524,288 | 256 | 3 | 198.070 | 55.554 | 3.565x | 3–0 |
| 1,048,576 | 64 | 3 | 231.368 | 77.557 | 2.983x | 3–0 |
| 1,048,576 | 128 | 3 | 228.960 | 65.767 | 3.481x | 3–0 |
| 1,048,576 | 256 | 3 | 226.165 | 58.447 | 3.870x | 3–0 |
| 2,097,152 | 64 | 3 | 263.618 | 109.156 | 2.415x | 3–0 |
| 2,097,152 | 128 | 3 | 264.491 | 85.244 | 3.103x | 3–0 |
| 2,097,152 | 256 | 3 | 259.333 | 73.784 | 3.515x | 3–0 |
| 4,194,304 | 64 | 3 | 409.608 | 139.425 | 2.938x | 3–0 |
| 4,194,304 | 128 | 3 | 399.183 | 106.479 | 3.749x | 3–0 |
| 4,194,304 | 256 | 3 | 351.003 | 87.314 | 4.020x | 3–0 |
| 8,388,607 | 64 | 3 | 490.165 | 115.609 | 4.240x | 3–0 |
| 8,388,607 | 128 | 3 | 618.996 | 102.622 | 6.032x | 3–0 |
| 8,388,607 | 256 | 3 | 608.300 | 86.902 | 7.000x | 3–0 |

#### Replace

[PNG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-replace-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-replace-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 14 | 5.827 | 0.902 | 6.459x | 14–0 |
| 1,024 | 128 | 14 | 3.424 | 0.477 | 7.182x | 14–0 |
| 1,024 | 256 | 14 | 1.855 | 0.242 | 7.669x | 14–0 |
| 2,048 | 64 | 14 | 11.058 | 1.748 | 6.325x | 14–0 |
| 2,048 | 128 | 14 | 6.630 | 0.923 | 7.180x | 14–0 |
| 2,048 | 256 | 14 | 3.494 | 0.451 | 7.753x | 14–0 |
| 4,096 | 64 | 14 | 21.794 | 3.433 | 6.349x | 14–0 |
| 4,096 | 128 | 14 | 12.756 | 1.810 | 7.046x | 14–0 |
| 4,096 | 256 | 14 | 6.715 | 0.889 | 7.555x | 14–0 |
| 8,192 | 64 | 14 | 41.519 | 6.651 | 6.242x | 14–0 |
| 8,192 | 128 | 14 | 23.805 | 3.545 | 6.715x | 14–0 |
| 8,192 | 256 | 14 | 12.942 | 1.765 | 7.331x | 14–0 |
| 16,384 | 64 | 14 | 80.552 | 12.338 | 6.529x | 14–0 |
| 16,384 | 128 | 14 | 43.652 | 6.353 | 6.871x | 14–0 |
| 16,384 | 256 | 14 | 24.437 | 3.250 | 7.520x | 14–0 |
| 32,768 | 64 | 14 | 127.678 | 19.358 | 6.596x | 14–0 |
| 32,768 | 128 | 14 | 78.286 | 10.294 | 7.605x | 14–0 |
| 32,768 | 256 | 14 | 43.434 | 5.182 | 8.381x | 14–0 |
| 65,536 | 64 | 14 | 166.212 | 23.245 | 7.151x | 14–0 |
| 65,536 | 128 | 14 | 92.762 | 11.794 | 7.865x | 14–0 |
| 65,536 | 256 | 14 | 48.261 | 5.614 | 8.597x | 14–0 |
| 131,072 | 64 | 14 | 199.996 | 28.271 | 7.074x | 14–0 |
| 131,072 | 128 | 14 | 107.185 | 14.308 | 7.491x | 14–0 |
| 131,072 | 256 | 14 | 54.116 | 6.740 | 8.029x | 14–0 |
| 262,144 | 64 | 14 | 194.130 | 32.052 | 6.057x | 14–0 |
| 262,144 | 128 | 14 | 103.924 | 16.059 | 6.471x | 14–0 |
| 262,144 | 256 | 14 | 53.279 | 7.592 | 7.018x | 14–0 |
| 524,288 | 64 | 14 | 212.755 | 34.876 | 6.100x | 14–0 |
| 524,288 | 128 | 14 | 113.046 | 17.424 | 6.488x | 14–0 |
| 524,288 | 256 | 14 | 61.472 | 8.291 | 7.414x | 14–0 |
| 1,048,576 | 64 | 14 | 229.391 | 37.253 | 6.158x | 14–0 |
| 1,048,576 | 128 | 14 | 128.527 | 18.537 | 6.933x | 14–0 |
| 1,048,576 | 256 | 14 | 67.907 | 8.776 | 7.738x | 14–0 |
| 2,097,152 | 64 | 14 | 256.160 | 39.499 | 6.485x | 14–0 |
| 2,097,152 | 128 | 14 | 138.832 | 19.299 | 7.194x | 14–0 |
| 2,097,152 | 256 | 14 | 72.423 | 9.017 | 8.032x | 14–0 |
| 4,194,304 | 64 | 14 | 278.148 | 40.614 | 6.849x | 14–0 |
| 4,194,304 | 128 | 14 | 147.913 | 19.650 | 7.528x | 14–0 |
| 4,194,304 | 256 | 14 | 74.142 | 9.144 | 8.108x | 14–0 |
| 8,388,607 | 64 | 14 | 297.144 | 40.722 | 7.297x | 14–0 |
| 8,388,607 | 128 | 14 | 151.851 | 19.472 | 7.798x | 14–0 |
| 8,388,607 | 256 | 14 | 75.039 | 8.980 | 8.356x | 14–0 |

#### Split

[PNG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.png) ·
[SVG chart](docs/_static/benchmarks/cudf-api-split-throughput-sweep.svg) ·
[case-level CSV](docs/_static/benchmarks/cudf-api-split-throughput-data.csv)

| Rows | StringBytes | Cases | Regex IR (M rows/s) | cuDF (M rows/s) | Speedup | Regex IR–cuDF wins |
|---:|---:|---:|---:|---:|---:|:---:|
| 1,024 | 64 | 7 | 4.100 | 1.096 | 3.741x | 7–0 |
| 1,024 | 128 | 7 | 2.791 | 0.619 | 4.506x | 7–0 |
| 1,024 | 256 | 7 | 1.748 | 0.334 | 5.239x | 7–0 |
| 2,048 | 64 | 7 | 7.875 | 2.129 | 3.700x | 7–0 |
| 2,048 | 128 | 7 | 5.249 | 1.214 | 4.325x | 7–0 |
| 2,048 | 256 | 7 | 3.269 | 0.622 | 5.253x | 7–0 |
| 4,096 | 64 | 7 | 14.107 | 4.246 | 3.322x | 7–0 |
| 4,096 | 128 | 7 | 9.797 | 2.325 | 4.213x | 7–0 |
| 4,096 | 256 | 7 | 5.977 | 1.208 | 4.946x | 7–0 |
| 8,192 | 64 | 7 | 25.020 | 7.896 | 3.169x | 7–0 |
| 8,192 | 128 | 7 | 17.832 | 4.484 | 3.977x | 7–0 |
| 8,192 | 256 | 7 | 11.132 | 2.368 | 4.700x | 7–0 |
| 16,384 | 64 | 7 | 43.597 | 13.877 | 3.142x | 7–0 |
| 16,384 | 128 | 7 | 28.760 | 7.778 | 3.698x | 7–0 |
| 16,384 | 256 | 7 | 18.688 | 4.353 | 4.293x | 7–0 |
| 32,768 | 64 | 7 | 66.868 | 20.529 | 3.257x | 7–0 |
| 32,768 | 128 | 7 | 44.054 | 12.622 | 3.490x | 7–0 |
| 32,768 | 256 | 7 | 26.916 | 6.744 | 3.991x | 7–0 |
| 65,536 | 64 | 7 | 83.009 | 26.630 | 3.117x | 7–0 |
| 65,536 | 128 | 7 | 49.463 | 15.214 | 3.251x | 7–0 |
| 65,536 | 256 | 7 | 29.511 | 8.358 | 3.531x | 7–0 |
| 131,072 | 64 | 7 | 95.279 | 31.513 | 3.023x | 7–0 |
| 131,072 | 128 | 7 | 59.644 | 18.557 | 3.214x | 7–0 |
| 131,072 | 256 | 7 | 34.769 | 9.816 | 3.542x | 7–0 |
| 262,144 | 64 | 7 | 96.843 | 34.771 | 2.785x | 7–0 |
| 262,144 | 128 | 7 | 64.154 | 20.559 | 3.120x | 7–0 |
| 262,144 | 256 | 7 | 39.280 | 10.795 | 3.639x | 7–0 |
| 524,288 | 64 | 7 | 113.772 | 40.363 | 2.819x | 7–0 |
| 524,288 | 128 | 7 | 76.477 | 23.048 | 3.318x | 7–0 |
| 524,288 | 256 | 7 | 45.901 | 12.109 | 3.791x | 7–0 |
| 1,048,576 | 64 | 7 | 141.690 | 44.862 | 3.158x | 7–0 |
| 1,048,576 | 128 | 7 | 92.586 | 25.145 | 3.682x | 7–0 |
| 1,048,576 | 256 | 7 | 55.748 | 12.900 | 4.322x | 7–0 |
| 2,097,152 | 64 | 7 | 175.388 | 49.718 | 3.528x | 7–0 |
| 2,097,152 | 128 | 7 | 107.948 | 27.184 | 3.971x | 7–0 |
| 2,097,152 | 256 | 7 | 63.010 | 13.711 | 4.596x | 7–0 |
| 4,194,304 | 64 | 7 | 207.663 | 55.377 | 3.750x | 7–0 |
| 4,194,304 | 128 | 7 | 128.207 | 29.214 | 4.389x | 7–0 |
| 4,194,304 | 256 | 7 | 69.159 | 14.264 | 4.848x | 7–0 |
| 8,388,607 | 64 | 7 | 246.588 | 58.869 | 4.189x | 7–0 |
| 8,388,607 | 128 | 7 | 134.984 | 29.921 | 4.511x | 7–0 |
| 8,388,607 | 256 | 7 | 55.213 | 14.243 | 3.877x | 7–0 |

[All API measurements](docs/_static/benchmarks/cudf-api-throughput-data.csv)

<!-- END GENERATED CUDF API RESULTS -->

The `regex_ir/cold` and `cudf/cold` registrations remain available for
one-shot setup measurements. Regex IR cold time includes parsing,
determinization, libNVVM compilation, PTX generation, device LTO, nvJitLink,
module loading, and the first launch. Production integrations should cache the
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

`Cold Compile` is recorded once during each state setup and is not included in
the execution latency above:

| Source suite | Regex IR cold JIT mean (ms) | cuDF program-create mean (ms) | Regex IR median (ms) | cuDF median (ms) |
|:---|---:|---:|---:|---:|
| OpenResty | 12.447 | 0.0162 | 10.867 | 0.0076 |
| Rust Leipzig | 17.468 | 0.0232 | 10.934 | 0.0094 |
| Boost/GCC | 17.723 | 0.0391 | 11.582 | 0.0108 |
| mariomka | 10.839 | 0.0117 | 10.840 | 0.0112 |
| **All imported suites** | **15.578** | **0.0269** | **11.066** | **0.0091** |

These setup columns intentionally measure different work. Regex IR's value is
a true device-code cold path—parse/lower, render NVVM IR, libNVVM `-opt=3`,
nvJitLink device LTO at `-O3`, and module load. cuDF creates an interpreter
program and does not JIT a specialized CUDA kernel.

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
| OpenResty | 31 | 52.598 | 2.716 | 19.366x | 31–0 | 12.365 | 0.0162 |
| Rust Leipzig | 18 | 40.440 | 1.947 | 20.770x | 18–0 | 17.702 | 0.0235 |
| Boost/GCC | 22 | 2.720 | 0.266 | 10.211x | 22–0 | 21.368 | 0.0560 |
| mariomka | 3 | 24.844 | 0.987 | 25.180x | 3–0 | 10.880 | 0.0124 |
| **All complete-corpus suites** | **74** | **19.840** | **1.205** | **16.459x** | **74–0** | **16.280** | **0.0297** |

Regex IR won 74 of 74 paired full-corpus cases. The narrowest result was Boost/GCC case
8 (2.464x); the largest was Boost/GCC case 14 (675.305x).

#### OpenResty full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-openresty-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-openresty-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | literal miss | 25.000 | 0.366 | 1.071 | 2.927x |
| 2 | short alt miss | 10.000 | 0.152 | 2.529 | 16.586x |
| 3 | suffix alt miss | 25.000 | 0.378 | 8.508 | 22.498x |
| 4 | suffix alt prose | 19.117 | 0.411 | 9.063 | 22.064x |
| 5 | wide class miss | 25.000 | 0.354 | 3.629 | 10.237x |
| 6 | split class miss | 25.000 | 0.357 | 3.691 | 10.330x |
| 7 | split class prose | 19.117 | 0.235 | 2.328 | 9.899x |
| 8 | large alt prose | 19.117 | 0.560 | 32.788 | 58.553x |
| 9 | large alt miss | 25.000 | 0.371 | 31.798 | 85.679x |
| 10 | nested alt | 25.000 | 0.368 | 8.078 | 21.940x |
| 11 | long nested alt miss | 10.000 | 0.173 | 8.598 | 49.793x |
| 12 | capture chain miss | 25.000 | 0.369 | 8.927 | 24.211x |
| 13 | capture chain random miss | 10.000 | 0.175 | 8.558 | 49.020x |
| 14 | lazy class repeat | 10.000 | 0.144 | 1.352 | 9.369x |
| 15 | lazy dot repeat | 10.000 | 0.145 | 1.344 | 9.241x |
| 16 | greedy dot repeat | 10.000 | 0.143 | 1.348 | 9.451x |
| 17 | anchored literal | 19.117 | 0.397 | 3.921 | 9.880x |
| 18 | literal prose | 19.117 | 0.367 | 1.755 | 4.779x |
| 19 | folded literal | 19.117 | 0.382 | 4.469 | 11.694x |
| 20 | class suffix | 19.117 | 0.409 | 5.935 | 14.507x |
| 21 | name alternation | 19.117 | 0.395 | 5.552 | 14.068x |
| 22 | word boundary | 19.117 | 0.443 | 9.255 | 20.901x |
| 23 | negated bounded | 19.117 | 0.482 | 14.870 | 30.845x |
| 24 | name literals | 19.117 | 0.432 | 8.833 | 20.437x |
| 25 | folded names | 19.117 | 0.489 | 10.751 | 21.980x |
| 26 | short prefix names | 19.117 | 0.430 | 27.909 | 64.974x |
| 27 | required prefix names | 19.117 | 0.435 | 28.745 | 66.005x |
| 28 | word suffix | 19.117 | 0.370 | 7.711 | 20.835x |
| 29 | bounded word suffix | 19.117 | 0.472 | 12.227 | 25.889x |
| 30 | captured name suffix | 19.117 | 0.439 | 9.507 | 21.679x |
| 31 | quoted sentence | 19.117 | 0.397 | 14.296 | 36.001x |

#### Rust Leipzig full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-leipzig-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-leipzig-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | twain | 15.272 | 0.297 | 1.607 | 5.402x |
| 2 | twain ignore case | 15.272 | 0.311 | 3.716 | 11.963x |
| 3 | shing | 15.272 | 0.335 | 4.949 | 14.757x |
| 4 | huck saw | 15.272 | 0.318 | 4.927 | 15.469x |
| 5 | word nn | 15.272 | 0.355 | 7.504 | 21.146x |
| 6 | negated bounded | 15.272 | 0.389 | 12.541 | 32.279x |
| 7 | names | 15.272 | 0.351 | 7.778 | 22.164x |
| 8 | names ignore case | 15.272 | 0.404 | 9.228 | 22.847x |
| 9 | optional prefix | 15.272 | 0.356 | 24.224 | 68.048x |
| 10 | required prefix | 15.272 | 0.354 | 24.033 | 67.974x |
| 11 | tom river | 15.272 | 2.070 | 13.380 | 6.464x |
| 12 | word ing | 15.272 | 0.304 | 6.339 | 20.817x |
| 13 | bounded ing | 15.272 | 0.388 | 10.092 | 26.022x |
| 14 | name suffix | 15.272 | 0.363 | 7.818 | 21.545x |
| 15 | quoted sentence | 15.272 | 0.326 | 13.663 | 41.864x |
| 16 | unicode symbols | 15.272 | 0.273 | 3.693 | 13.513x |
| 17 | math symbol property | 15.272 | 0.275 | 2.913 | 10.575x |
| 18 | csv field | 15.272 | 0.301 | 14.634 | 48.582x |

#### Boost/GCC full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-boost-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-boost-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | long twain | 19.117 | 0.368 | 1.773 | 4.812x |
| 2 | long huck | 19.117 | 0.366 | 2.041 | 5.581x |
| 3 | long ing | 19.117 | 0.371 | 7.786 | 20.987x |
| 4 | long line twain | 19.117 | 0.401 | 7.357 | 18.337x |
| 5 | long names | 19.117 | 0.433 | 9.063 | 20.931x |
| 6 | long names near river | 19.117 | 2.755 | 27.632 | 10.031x |
| 7 | medium twain | 0.048 | 0.032 | 0.087 | 2.690x |
| 8 | medium huck | 0.048 | 0.032 | 0.078 | 2.464x |
| 9 | medium ing | 0.048 | 0.030 | 0.344 | 11.413x |
| 10 | medium line twain | 0.048 | 0.031 | 0.343 | 10.962x |
| 11 | medium names | 0.048 | 0.033 | 0.285 | 8.664x |
| 12 | medium names near river | 0.048 | 0.166 | 0.635 | 3.812x |
| 13 | cpp declaration | 0.033 | 0.034 | 0.340 | 10.013x |
| 14 | cpp tokens | 0.033 | 0.122 | 82.473 | 675.305x |
| 15 | cpp include | 0.033 | 0.030 | 0.227 | 7.483x |
| 16 | boost include | 0.033 | 0.033 | 0.234 | 7.020x |
| 17 | html names | 0.049 | 0.039 | 0.285 | 7.390x |
| 18 | html paragraph | 0.049 | 0.036 | 0.287 | 8.072x |
| 19 | html anchor | 0.049 | 0.039 | 0.774 | 19.858x |
| 20 | html heading | 0.049 | 0.036 | 0.304 | 8.453x |
| 21 | html image | 0.049 | 0.039 | 0.426 | 10.937x |
| 22 | html font | 0.049 | 0.040 | 0.275 | 6.852x |

#### mariomka full-corpus cases

[PNG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.png) ·
[SVG chart](docs/_static/benchmarks/corpus-mariomka-throughput-cases.svg) ·
[case-level CSV](docs/_static/benchmarks/corpus-mariomka-throughput-data.csv)

| Case | Expression role | Input (MiB) | Regex IR (ms) | cuDF (ms) | Speedup |
|---:|:---|---:|---:|---:|---:|
| 1 | email | 6.523 | 0.251 | 3.875 | 15.467x |
| 2 | uri | 6.523 | 0.262 | 6.935 | 26.429x |
| 3 | ipv4 | 6.523 | 0.256 | 10.014 | 39.057x |

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

The generated NVVM now selects among five executor strategies:

- a direct single-byte scan for exact ASCII global count, replacement, and
  split expressions;
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

Cold latency still includes parsing, determinization, libNVVM compilation, PTX
generation, device LTO, nvJitLink, module loading, and the first launch. cuDF
copies a compact predefined instruction program instead of invoking a device
compiler, so it remains faster for uncached one-shot use. Production users
should cache the linked cubin by pattern, options, architecture, and toolkit
version; steady-state execution is the path where JIT specialization pays off.

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
