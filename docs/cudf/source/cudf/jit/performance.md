
# JIT Performance Guidelines

## When to Use JIT

### 1. Dynamic Expressions

Kernels that are generated from dynamic expressions (e.g. SQL expressions, UDFs, etc.) that are not known at compile time can benefit from JIT compilation. This is because the kernel code can be specialized at runtime for the specific expression and input types, which can lead to significant performance improvements.  This is especially important for expressions that are executed multiple times, as the overhead of JIT compilation can be amortized over many executions.

Each expression has different arithmetic intensity and memory access patterns, which can be optimized by the JIT compiler. Static expressions that are known at compile time will most benefit from standard AOT kernel compilation.

libcudf also has a pre-compiled AST interpreter that can be used to evaluate expressions without JIT compilation. Its performance is generally lower than JIT compilation but provides lower cold-start latency for single-shot expression evaluation. It is limited by:
- The types of expressions it can evaluate
- The types of inputs it can handle
- The dimensionality of the inputs it can handle (scalar vs columnar)
- Register pressure and memory bandwidth usage for complex expressions
- Intermediate results that are materialized in shared memory instead of registers, which can lead to lower performance for complex expressions
- Increased AOT compilation times, thus increasing the overall build time of libcudf and its dependencies. This is because the pre-compiled AST interpreter needs to be compiled for all possible expressions and input types, which can lead to a combinatorial explosion of code paths and makes the compiler work harder to optimize the code.


### 2. Architecture-Specific Optimizations & Optimization Policies

For workloads that require very specific optimization policies (e.g. memory layouts, vectorization policies, kernel configurations, etc.) that are not known at compile time (i.e. GPU-specific tuning policies), JIT compilation can be used to generate specialized kernels that are optimized for the specific workload. Pre-compiling kernels for all possible configurations is not feasible due to the combinatorial explosion of code paths and the increased AOT compilation times. JIT compilation allows for the generation of specialized kernels at runtime, which can lead to better codegen and significant performance improvements.


### 3. Long-Running High-Throughput Workloads

JIT compilation is most beneficial for long-running, high-throughput workloads where the overhead of JIT compilation can be amortized over many executions. This is because the JIT compiler can generate specialized code that is optimized for the specific workload, which can lead to significant performance improvements.
The overhead of JIT compilation can be significant, especially for complex expressions, so it is important to consider the trade-offs between JIT compilation and pre-compiled kernels when designing workloads. For example, compiling a transform kernel with cold PCH cache can take up to 900ms, while compiling the same kernel with hot PCH cache can take 80ms. This is a significant difference that can impact the overall performance of a workload. We continue to optimize the JIT compilation path to reduce this overhead, but it is important to consider the trade-offs between JIT compilation and pre-compiled kernels when designing workloads.

For workloads that require low cold-start latency, pre-compiled kernels may be preferred over JIT compilation.


## Performance Considerations

When benchmarking JIT workloads, it is important to consider the following factors:

- **Cold vs Hot Runs**: Cold runs are JIT compilations that have not been compiled and cached, while hot runs are JIT compilations that have already been JIT-compiled and cached. Cold runs can be significantly slower than hot runs due to the overhead of JIT compilation, which includes parsing and resolving C++ code, generating PTX code, and compiling the PTX code into machine code. It is important to consider if the compilation latency of the cold run is acceptable for the workload, or if the workload is long-running enough to amortize the overhead of JIT compilation over many executions. Benchmarking JIT workloads should include both cold and hot runs to understand the performance characteristics of the workload.

- **Latency Requirements**: Some workloads may have strict latency requirements that cannot tolerate the overhead of JIT compilation. In these cases, pre-compiled kernels (or the AST interpreter) may be preferred over JIT compilation. It is important to consider the latency requirements of the workload when benchmarking JIT workloads. A pre-warmed JIT kernel can be used to reduce the cold-start latency of JIT compilation, but it may not be feasible for all workloads due to the combinatorial explosion of code paths and the increased AOT compilation times. It is important to consider the trade-offs between JIT compilation and pre-compiled kernels when designing workloads.

- **Arithmetic Intensity & Intermediate Count**: The arithmetic intensity of the workload is an important factor to consider when benchmarking JIT workloads. Workloads with high arithmetic intensity (i.e. a high ratio of arithmetic operations to memory operations) can benefit more from JIT compilation, as the overhead of JIT compilation can be amortized over many executions. Workloads with low arithmetic intensity may not benefit as much from JIT compilation. The number of intermediate results that are generated during expression evaluation impacts the performance of workloads. For executing arbitrary expressions whilst avoiding materializing intermediate results in global memory, cuDF has a pre-compiled AST interpreter and a JIT compiler. To reduce global memory bandwidth usage, the AST interpreter materializes intermediate results in shared memory, for workloads with a low number of intermediates, the cost of shared memory usage will be insignificant, and the AST interpreter will perform well. But with an expression that has a high number of intermediates, the JIT compiler can generate specialized code that avoids materializing intermediate results in shared memory and purely uses registers, which can lead to significant performance improvements.

- **Amortization of JIT Compilation Overhead**: The overhead of JIT compilation can be significant, especially for complex expressions and kernels. It is important to consider the break-even point where the overhead of JIT compilation is amortized over many executions. This amortization point primarily depends on the number of rows to be processed and the number of times the kernel is re-used.


<!-- TODO -->

## How Precompiled Headers Affect JIT Performance

libcudf uses precompiled headers (PCH) to speedup CUDA JIT compilation. Precompiled headers are fully resolved ASTs of the CUDA C++ headers, this helps to eliminate the overhead of parsing and resolving C++ code during JIT compilation. The specifics of this representation is internal to the compiler (NVRTC) and is not exposed to the user nor portable across different compilers.

Precompiled headers are presently generated at **the first** cold JIT compilation of a kernel, so the first JIT compilation request of a kernel always uses a cold PCH cache. The precompiled headers are cached and reused for subsequent cold JIT compilations of kernels. This means that the first cold JIT compilation of a kernel is always slower than subsequent cold JIT compilations of the same kernel because it pays the full CUDA C++ frontend compilation cost.

<img src="https://private-user-images.githubusercontent.com/26050398/551726508-08e7f29f-6d0c-4936-98f3-71a70ad95f4b.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODI3MDM5NzYsIm5iZiI6MTc4MjcwMzY3NiwicGF0aCI6Ii8yNjA1MDM5OC81NTE3MjY1MDgtMDhlN2YyOWYtNmQwYy00OTM2LTk4ZjMtNzFhNzBhZDk1ZjRiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA2MjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNjI5VDAzMjc1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY1MjQ4MGFiYzJhMjk0ZTNkMjQ1NzBlNTQ4NWUzZTA2NTEzZDg5ZDg5NjdjNTc5MDQwYzkzZjhkNTFhOTRmNmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.aH8C2EUYszUF49I1EHZhr3qXoEUJqaFEFFxcIzwu168"/>

This is because NVRTC's PCH cache is **process-local** and not shared across processes. This means that if a kernel is compiled in one process, the precompiled headers are not available to other processes. This can lead to performance degradation in multi-process environments where the same kernel is compiled in different processes. If NVRTC's PCH could be prepared at compile time, this would eliminate the cold PCH cache overhead entirely.



## JIT Performance Tuning Options

### `LIBCUDF_KERNEL_CACHE_PATH:path`

This environment variable affects where the libcudf JIT kernel cache path is located. For a multi-process container environment (i.e. Presto multi-container executor) the paths should point to a shared location to ensure re-use of compiled kernels across the processes. If this path is not specified, libcudf will attempt to find a suitable and accessible filesystem path to store compilation artifacts/kernels.


### `LIBCUDF_JIT_ENABLED:bool`

This environment variable affects various libcudf subsystems. It informs the subsystems to prefer JIT for expression evaluation. For example, it informs the Parquet Reader post-filter subsystem to prefer JIT for evaluating expressions. It also informs `compute_column` and other functions that accept an `AST` tree to prefer JIT for evaluating expressions which can give significant performance improvements with high arithmetic intensity expressions.

### `LIBCUDF_KERNEL_CACHE_PRELOAD:bool`

This environment variable determines if libcudf should preload all previously compiled kernels from the kernel cache path into memory (`cuLibrary`). This can be helpful to measure performance for hot benchmark runs and make sure there's no overhead from loading the kernel.

### `LIBCUDF_KERNEL_CACHE_DISABLED:bool`

This environment variable disables caching of libcudf's kernel artifacts. This is useful for benchmarking in that it helps to ensure JIT compilation always goes through the cold cache path. For purely cold kernel cache measurements this should be used with the `LIBCUDF_JIT_DISABLE_CUDA_CACHE` to disable cuDF's CUDA codegen cache.

### `LIBCUDF_KERNEL_CACHE_CLEAR:bool`

This environment variable removes all cached kernel artifacts in libcudf's JIT kernel cache path.

### `LIBCUDF_JIT_DISABLE_CUDA_CACHE:bool`

This environment variable disables cuDF's CUDA codegen cache. It affects PTX and CUBIN code generation after CUDA frontend processing.


### `LIBCUDF_JIT_VERBOSE:bool`

This environment variable informs libcudf to print verbose information about JIT compilation. This is useful for debugging and benchmarking purposes. It prints information such as:
- NVRTC compilation times (CUDA C++ frontend, PTXAS, Codegen)
- NVRTC include headers
- NVRTC compilation flags

### `LIBCUDF_JIT_DUMP_TRACE:bool`

This environment variable enables dumping of JIT compilation traces. It is useful for debugging and performance analysis. The trace is dumped to the command line as text
- NVRTC compilation times (CUDA C++ frontend, PTXAS, Codegen)
- Optimization passes and their times
- NVRTC passes and their times


### `LIBCUDF_JIT_DUMP_TIME_PROFILE:bool`

This is same as `LIBCUDF_JIT_DUMP_TRACE` but dumps the trace as a perfetto json trace file. The trace file is named dump_jit_time_profile `libcudf_kernel_${kernel_name}_trace.json` (e.g. `libcudf_kernel_transform_trace.json`, `libcudf_kernel_filter_join_indices_trace.json`) and is dumped to the current working directory. The trace file can be opened in Chrome's tracing tool or in Perfetto UI.

### `LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS:integer`

This environment variable limits the number of cached kernels per process. This is useful to control the memory footprint of the kernel cache in multi-process environments. It can also be used to disable caching of kernels for benchmarking purposes.


### `cudf::enable_jit_cache(bool)`

This function enables or disables the libcudf JIT kernel cache. It is useful for benchmarking in that it helps to ensure JIT compilation always goes through the cold cache path. For purely cold kernel cache measurements this should be used with the `LIBCUDF_JIT_DISABLE_CUDA_CACHE` to disable cuDF's CUDA codegen cache. This is analogous to the `LIBCUDF_KERNEL_CACHE_DISABLED` environment variable but can be used programmatically.

### `cudf::enable_cuda_cache(bool)`

This function enables or disables cuDF's CUDA codegen cache. It is useful for benchmarking in that it helps to ensure JIT compilation always goes through the cold cache path. For purely cold kernel cache measurements this should be used with the `LIBCUDF_JIT_DISABLE_CUDA_CACHE` to disable cuDF's CUDA codegen cache. This is analogous to the `LIBCUDF_JIT_DISABLE_CUDA_CACHE` environment variable but can be used programmatically.

### `cudf::clear_jit_cache()`

This function removes all cached kernel artifacts in libcudf's JIT kernel cache path. It is useful for benchmarking in that it helps to ensure JIT compilation always goes through the cold cache path. For purely cold kernel cache measurements this should be used with the `cudf::enable_cuda_cache(false)` to disable cuDF's CUDA codegen cache. This is analogous to the `LIBCUDF_KERNEL_CACHE_CLEAR` environment variable but can be used programmatically.
