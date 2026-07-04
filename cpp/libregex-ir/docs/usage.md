# Using Regex IR

Regex IR compiles a regular expression on the host and emits a self-contained
NVVM module for NVIDIA device code. It does not provide a host JIT or a public
runtime matcher, and it does not claim to emit target-independent LLVM IR. The
interpreter exercised by `tests/unit_tests.cpp` is a correctness oracle, not an
installed execution API.

The production path is:

    host regex compilation
      -> optimized Instruction IR
      -> textual NVVM IR
      -> libNVVM PTX
      -> nvJitLink with kernel LTO IR
      -> load and launch cubin

## 1. Link or vendor the compiler

An installed package exports `regex_ir::regex_ir`:

```cmake
find_package(regex_ir CONFIG REQUIRED)

add_executable(regex_codegen codegen.cpp)
target_link_libraries(regex_codegen PRIVATE regex_ir::regex_ir)
target_compile_features(regex_codegen PRIVATE cxx_std_20)
```

The library can also be vendored as the two files in the repository root:

```cmake
find_package(fmt CONFIG REQUIRED)
add_library(regex_ir path/to/regex_ir.cpp)
target_include_directories(regex_ir PUBLIC path/to)
target_compile_features(regex_ir PUBLIC cxx_std_20)
target_link_libraries(regex_ir PRIVATE fmt::fmt)
```

Include the public API with:

```cpp
#include <regex_ir.hpp>
```

The core library requires neither CUDA nor LLVM. It uses libfmt's `fmt::format`
and requires a C++20 host compiler.

## 2. Compile a pattern

`compile()` parses, builds ordered Thompson Automata IR, lowers to typed
Instruction IR, optimizes, and verifies the result.

```cpp
regex_ir::compile_options options;
options.case_insensitive = false;
options.multiline = false;
options.dot_all = false;
options.characters = regex_ir::character_mode::UTF8;

auto result = regex_ir::compile(
  "abc[0-9]+", regex_ir::operation::contains(), options);

if (!result) {
  for (auto& diagnostic : result.diagnostics) {
    std::cerr << "pattern byte " << diagnostic.span.offset << ": "
              << diagnostic.message << '\n';
  }
  return;
}

auto& ir = *result.value;
```

Select the operation before lowering because it controls search policy,
end-of-input requirements, capture retention, and result shape:

| Construction | Result policy | Matching policy |
| --- | --- | --- |
| `operation::matches()` | Boolean | Attempt at byte zero and require end-of-input. |
| `operation::contains()` | Boolean | Search for the first leftmost match. |
| `operation::find()` | Match span | Return the first leftmost match. |
| `operation::extract()` | Capture spans | Return the first match and captures. |
| `operation::count()` | Count | Count non-overlapping leftmost matches. |
| `operation::replace(text)` | Replacement | Replace non-overlapping matches. |
| `operation::split()` | Fields | Split at non-overlapping matches. |

NVVM IR generation accepts the two boolean result shapes and the capture result
shape produced by `operation::extract()`. Count, findall, replacement, and split
can be materialized around repeated capture-enumeration calls, as demonstrated
by the `Cudf` fixture in `tests/unit_tests.cpp`. The host interpreter directly
materializes every result shape.

For stage-by-stage inspection:

```cpp
auto automata = regex_ir::compile_automata(pattern, options);
auto lowered = regex_ir::lower(*automata.value, regex_ir::operation::contains());
auto optimized = regex_ir::optimize(std::move(*lowered.value));
```

Call `verify()` before processing externally modified IR. `to_string()` is a
deterministic diagnostic printer, not a serialization format.

## 3. Generate NVVM IR

```cpp
regex_ir::nvvm_ir_codegen_options codegen;
codegen.symbol_prefix = "tenant_17";
codegen.execute_function = "regex_contains_17";
codegen.prefix_filter = true;
codegen.branch_hints = true;

auto nvvm_ir = regex_ir::generate_nvvm_ir(ir, codegen);
```

Both names must match `[A-Za-z_][A-Za-z0-9_]*` and must not use reserved LLVM
or NVVM prefixes. `symbol_prefix` is applied to every generated helper and the
block dispatcher. `execute_function` is the public device function referenced
by the kernel. Using a unique prefix and execute name lets independently
generated modules coexist in one link scope.

Input bytes use ordinary NVVM loads, leaving cache selection to the CUDA
toolchain. When the recursive fallback is selected, `prefix_filter` skips
impossible scan positions if the entry block proves a required ASCII literal
or singleton, and `branch_hints` adds the NVVM-supported `llvm.expect`
intrinsic to the resulting biased candidate branch. Deterministic modules do
not use either option.

The generated boolean entry ABI is:

```llvm
define zeroext i1 @regex_contains_17(i8* %data, i64 %size)
```

- `data` is a device pointer to one string's bytes.
- `size` is its byte length.
- the `i1` result reports whether contains or matches succeeded.
- UTF-8 mode advances by decoded code points while all positions remain byte
  offsets.
- byte mode treats every byte as one character.

An IR compiled with `operation::extract()` instead uses:

```llvm
define zeroext i1 @regex_extract_17(
  i8* %data, i64 %size, i64 %search_start, i64* %captures)
```

- `search_start` is the first byte position considered for the next match.
- `captures` has `2 * (capture_count + 1)` entries.
- entries zero and one receive the whole match's begin/end byte offsets.
- each numbered capture receives its begin/end pair; an unmatched optional
  capture remains `-1, -1`.
- the function returns false when no match exists at or after `search_start`.

Advance the next search to the previous end. After an empty match, advance by
one decoded character; after an empty match at end of input, stop. This is the
same enumeration policy used by count, findall, replacement, and split.

The module contains its own byte/UTF-8 decoding, predicates, assertions,
literal matching, prioritized control flow, NVPTX data layout, and NVVM IR 2.0
metadata. It has no external runtime-helper dependency.

The generator uses direct text concatenation with `fmt::format`; it does not
use or link LLVM IRBuilder.

## 4. How Instruction IR maps to generated code

Instruction blocks are dense control-flow nodes. Instructions execute in
storage order, and successor edges are attempted in ascending priority.
Priority preserves left-to-right alternation and greedy/lazy behavior.

| Instruction | Generated NVVM behavior |
| --- | --- |
| `can_peek{n}` | Check that `n` logical characters can be decoded at the cursor; fail the path otherwise. |
| `read_character` | No separate load is needed; the following predicate helper decodes at the current cursor. |
| `match_character{p}` | Decode one code point and test normalized ranges, negation, or dot/newline behavior. |
| `match_literal{s}` | Compare fused ASCII literals directly as bytes with one bounds check; otherwise decode and compare code points. |
| `advance_cursor{n}` | Advance by integer byte addition after a proven ASCII literal, or by `n` decoded characters otherwise. |
| `test_assertion{a}` | Test absolute/line anchors or configured word-boundary state without consuming input. |
| `write_capture` | Omitted for boolean operations after capture-stripping optimization. |
| `emit_accept` | Return true, additionally checking `position == size` for matches mode. |

For boolean contains and matches, captures and priority are unobservable. When
the graph has no position-dependent assertions, the generator computes its
deterministic states and Unicode predicate-equivalence classes at generation
time. The emitted function streams each row once through constant class and
transition tables. Contains includes the initial closure after every character;
matches starts only at byte zero and tests acceptance at end-of-input.

Assertion-bearing graphs retain the recursive Thompson fallback. In that path,
an ambiguous block saves its cursor, calls successors in priority order, and
restores the cursor after a failed branch. A generated step budget bounds
nullable cycles. The generated header identifies the selected executor and,
for deterministic output, reports its state and alphabet-class counts.

## 5. Compile generated IR with libNVVM

The integration uses the libNVVM C API:

```cpp
nvvmProgram program{};
nvvmCreateProgram(&program);
nvvmAddModuleToProgram(
  program, nvvm_ir.data(), nvvm_ir.size(), "generated_regex.nvvm");

std::string architecture_option = "-arch=" + compute_architecture;
char const* verify_options[] = {architecture_option.c_str()};
nvvmVerifyProgram(program, 1, verify_options);

char const* compile_options[] = {architecture_option.c_str(), "-opt=3"};
nvvmCompileProgram(program, 2, compile_options);

std::size_t ptx_size{};
nvvmGetCompiledResultSize(program, &ptx_size);
std::string ptx(ptx_size, '\0');
nvvmGetCompiledResult(program, ptx.data());
nvvmDestroyProgram(&program);
```

Check every return value. On verify or compile failure, retrieve the diagnostic
text with `nvvmGetProgramLogSize` and `nvvmGetProgramLog`. Choose a compute
architecture supported by the installed toolkit and compatible with the final
link target. Derive `compute_architecture` from the selected device's compute
capability instead of baking a device generation into the application. `-opt=3`
is libNVVM's documented default, but passing it explicitly makes the intended
optimization level visible and testable.

## 6. Provide and precompile the CUDA kernel

The surrounding kernel declares the selected execute name as an external
device function. A one-thread-per-string contains kernel can use this ABI:

```cpp
#include <cstddef>
#include <cstdint>

extern "C" __device__ bool regex_contains_17(char const*, std::size_t);

extern "C" __global__ void contains_kernel(char const* chars,
                                             std::int32_t const* offsets,
                                             std::int32_t rows,
                                             std::uint8_t* result)
{
  auto row = static_cast<std::int32_t>(
    blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;

  auto begin = offsets[row];
  auto end = offsets[row + 1];
  result[row] = regex_contains_17(
    chars + begin, static_cast<std::size_t>(end - begin));
}
```

Compile this stable kernel to an LTO-IR fatbin ahead of time, then embed it in
the host executable with the CUDA Toolkit's `bin2c` utility:

```console
nvcc --fatbin --dlink-time-opt --gen-opt-lto \
  --relocatable-device-code=true -std=c++20 \
  contains_kernel.cu -o contains_kernel.fatbin
bin2c --const --length --name contains_kernel_fatbin \
  contains_kernel.fatbin > contains_kernel.fatbin.inc
```

The declaration name and function signature must exactly match
`execute_function` and the generated ABI.

## 7. Link PTX and kernel LTO IR with nvJitLink

Create a linker for the real device architecture and enable LTO:

```cpp
std::string architecture_option = "-arch=" + sm_architecture;
char const* options[] = {architecture_option.c_str(), "-lto", "-O3"};
nvJitLinkHandle linker{};
nvJitLinkCreate(&linker, 3, options);

nvJitLinkAddData(linker,
                 NVJITLINK_INPUT_FATBIN,
                 contains_kernel_fatbin,
                 contains_kernel_fatbinLength,
                 "contains_kernel.fatbin");
nvJitLinkAddData(linker,
                 NVJITLINK_INPUT_PTX,
                 ptx.data(),
                 ptx.size(),
                 "generated_regex.ptx");
nvJitLinkComplete(linker);

std::size_t cubin_size{};
nvJitLinkGetLinkedCubinSize(linker, &cubin_size);
std::vector<char> cubin(cubin_size);
nvJitLinkGetLinkedCubin(linker, cubin.data());
nvJitLinkDestroy(&linker);
```

`sm_architecture` must be derived from the CUDA device selected for execution.
`-lto` enables device link-time optimization and `-O3` requests the highest
documented nvJitLink optimization level explicitly. Retrieve the nvJitLink
error log whenever an operation fails. The most common integration failures
are an execute-name mismatch, an ABI mismatch, or incompatible architecture
options.

## 8. Load and execute the linked kernel

Use the CUDA Driver API to load the cubin and launch the kernel:

```cpp
CUmodule module{};
CUfunction kernel{};
cuModuleLoadData(&module, cubin.data());
cuModuleGetFunction(&kernel, module, "contains_kernel");

void* arguments[] = {&chars, &offsets, &rows, &result};
cuLaunchKernel(kernel,
               grid_x, 1, 1,
               block_x, 1, 1,
               0, stream,
               arguments, nullptr);
```

Allocate and populate the buffers according to the kernel ABI, synchronize
according to the application's stream policy, and unload the module when its
cached lifetime ends. The current NVVM executor uses recursive calls for
prioritized graph traversal only when deterministic lowering is unavailable.
If the generated header reports `executor: recursive Thompson`, configure and
validate enough CUDA per-thread stack for the largest supported pattern and
row. Stack-limit configuration is context wide and should happen outside
latency-sensitive execution. Deterministic modules use CUDA's default stack.

Cache compiled artifacts by pattern, operation, compile options, optimization
options, NVVM code-generation options, Regex IR version, CUDA toolkit version,
and target architecture. Excluding any of these can reuse incompatible code.

The complete compile/link and launch sequence is tested in
`tests/unit_tests.cpp`; its stable CUDA entry points are in
`tests/regex_ir_execute_kernel.cu` and `tests/regex_ir_capture_kernel.cu`. The
optional GPU benchmark in `benchmarks/gpu_benchmark.cpp` uses an equivalent
libNVVM/nvJitLink path with a benchmark-specific CUDA wrapper.

## 9. Command-line tools

Generate NVVM IR directly:

```console
regex-ir-codegen \
  --operation contains \
  --symbol-prefix tenant_17 \
  --execute-function regex_contains_17 \
  'abc[0-9]+' > regex.nvvm
```

Inspect the pipeline interactively or for one pattern:

```console
regex-ir-explorer --nvvm --ir 'abc[0-9]+'
regex-ir-explorer --help
```

The explorer can independently show Automata IR plus its ASCII graph,
optimized Instruction IR, and generated NVVM IR.

## 10. Host interpreter for testing

`regex_ir::testing::execute()` and `regex_ir::testing::enumerate()` interpret
optimized Instruction IR on the CPU. They implement captures and every
operation result shape for semantic tests, fuzzing, and the RE2 CPU benchmark.
They deliberately use recursive control flow and dynamic containers and are not
a production host runtime.

When extending IR or NVVM IR generation:

1. Add semantic cases against the interpreter.
2. Compare optimized and optimization-disabled interpreter results.
3. Add deterministic NVVM-text assertions where useful.
4. Add representative modules to the libNVVM verification test.
5. Add a linked-kernel or GPU differential test when the ABI changes.

The default fuzz smoke test covers arbitrary parser inputs, exhaustive small
expressions, optimizer equivalence, verifier mutations, and all interpreter
operations. `tests/fuzz_tests.cpp` also provides the open-ended
ASan/UBSan/libFuzzer entry point.
