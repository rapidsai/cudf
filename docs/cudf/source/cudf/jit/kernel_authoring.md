

## CUDA Kernel Authoring for JIT Compilation

This document explains the mental model for how to develop CUDA kernels for JIT compilation in libcudf. It is intended for developers who are familiar with CUDA programming and want to write custom kernels that can be JIT compiled at runtime.

Kernel specialization in cuDF is of two types:
- Kernel Configuration Specialization: this determines the physical shape of the kernel, i.e: memory layout, nullability, vectorization, input types, output types, etc. It directly affects bandwidth usage, memory access patterns, and register usage of the kernel.
- Kernel User-Defined-Function (UDF) Specialization: this determines the logical behaviour of the kernel, i.e: the operations performed, the computation logic, etc. It directly affects the number of instructions, instruction-level parallelism, register pressure, and instruction-level optimizations of the kernel.

### 1. Kernel Template Construction

The kernel source template is a CUDA source code which is templatized to allow for kernel instantiation and specialization.

An illustrative kernel source template for a binary operation kernel that takes a single input column and produces a single output column is shown below:

```cpp

// file: binary_op_kernel.cu
#include <cudf/detail/kernel_instance.cuh> // runtime-specializes the `CUDF_KERNEL_INSTANCE` macro
#include <cudf/detail/operation_udf.cuh> // runtime-specializes the `GENERIC_TRANSFORM_OP` macro

template<typename LhsType, typename RhsType, bool RhsIsScalar, typename OutputType>
__device__ void binary_op_kernel(column_device_view const * inputs, mutable_column_device_view const * output, size_type n){
       auto start  = detail::grid_1d::global_thread_id();
       auto stride = detail::grid_1d::grid_stride();

        for (auto i = start; i < n; i += stride) {
            GENERIC_TRANSFORM_OP(&output->element<OutputType>(i), inputs[0]->element<LhsType>(i), inputs[1]->element<RhsType>(i * RhsIsScalar));
        }
}


extern "C" __device__ void cudf_kernel_entry(column_device_view const * inputs, mutable_column_device_view const * outputs, size_type n){
    CUDF_KERNEL_INSTANCE(inputs, outputs, n);
}

```

A few things to note:

- The kernel template `binary_op_kernel` is templatized on the input and output types, allowing for instantiation and specialization of the kernel across each input and output type combination.
- The `cudf_kernel_entry` function is the scope-independent entry point for the kernel, which is used to reference the kernel entry point and allows the entry point to be referenced independently of the kernel's template parameters as it is not tied to a complex mangling scheme.
- The `CUDF_KERNEL_INSTANCE` macro is used to instantiate/specialize the kernel for each input and output type combination and it is not known or resolved until runtime, this represents the kernel configuration specialization.
- The `GENERIC_TRANSFORM_OP` macro is used to instantiate/specialize the kernel for each UDF and it is not known or resolved until runtime, representing the kernel UDF specialization.


### 2. Source Embedding & Registration

The kernel source template above is embedded into the libcudf binary as a compressed string literal. This allows for the kernel source template to be specialized at runtime and compiled into a kernel for execution on the GPU. The kernel source templates are registered and embedded into libcudf through the CMake function `rtcx_embed_includes` provided by librtcx.

e.g.

```cmake

rtcx_embed_includes(
  cudf_cuda_embed # ID of target to embed the source template into
  SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_unary_op_kernel/jit # directory containing the kernel source template
  DEST_DIRECTORY cudf/cpp/src/sample_unary_op_kernel/jit # directory on the host where the source_directory will be installed to
  INCLUDE_DIRECTORIES cudf/cpp/src # include directory to be added for the target
)

```


### 3. Kernel Runtime Reflection & Specialization

At runtime, the kernel is instantiated and specialized by the user based on runtime reflection of runtime arguments. This means runtime values/arguments are used to determine the kernel's type and configuration.
This primarily involves mapping runtime input and output types to the kernel template's type parameters, which allows for the kernel to be specialized for the specific input and output types at runtime.

e.g.

<!-- construct mermaid diagram to better illustrate this -->

```cpp
// reflection based on runtime parameters
std::string reflect_binary_op_kernel(
    data_type lhs_type /* = type_id::INT32 */,
    data_type rhs_type /* = type_id::FLOAT32 */,
    bool rhs_is_scalar /* = false */, 
    data_type output_type /* = type_id::FLOAT32 */); 


// Kernel instance specialization (substituted during call to `cudf::get_udf_kernel`)
// #define CUDF_KERNEL_INSTANCE binary_op_kernel<int32_t, float, false, float>

```

Note: The mapping to the kernel instance can be more sophisticated and include policies like: optimal kernel configurations, memory layouts, vectorization policies, architecture-specific optimizations etc. which can be determined at runtime based on the input and output types and their properties.


### 4. UDF Specialization

At runtime, the User-Defined Function (UDF) is resolved to CUDA code and specializes the `GENERIC_TRANSFORM_OP` macro in the kernel template. This allows for the kernel's generated binary code to be specialized for the specific UDF at runtime, allowing for register & instruction optimizations and inlining of the UDF into the kernel's generated binary code.

e.g.

```cpp

// runtime UDF
__device__ void transform(float * out, int lhs, float rhs){
    *out = (lhs + rhs) * 0.5F;
}

// Kernel UDF specialization (substituted during call to `cudf::get_udf_kernel`)
// #define GENERIC_TRANSFORM_OP(...) transform(__VA_AGS__)

```

### 5. Kernel Execution

The compiled kernel can be launched on the GPU using the corresponding kernel handle and the kernel's arguments.
Given that the kernel uses a static entry point, it is recommended that the kernel's arguments are type-erased and non-dependent on the kernel's template parameters. This allows for the kernel to be launched independently of the kernel's template parameters.

```cpp
auto kernel_instance = "binary_op_kernel<int32_t, float, false, float>";
auto udf = "__device__ void transform(float * out, int lhs, float rhs){ *out = (lhs + rhs) * 0.5F; }";
auto kernel = cudf::get_udf_kernel("binary_op_kernel.cu", kernel_instance, udf);
column_device_view const * inputs = ...;
mutable_column_device_view const * outputs = ...;
size_type n = ...;
kernel.launch_with({grid_size}, {block_size}, 0, stream, inputs, outputs, n);
```


`cudf::get_udf_kernel` is responsible for generating the specialized CUDA kernel source code, compiling the kernel, caching it, and returning a handle (`cudf::kernel`) to the compiled kernel.

<!-- Generate NVIDIA-style diagram to illustrate the entire flow -->

NOTE: As of 26.06 PTX UDFs are supported by converting them to CUDA C++ code by using the `asm` CUDA directive. This still pays the full CUDA C++ frontend compilation cost.



## LTO JIT Model

The compilation model described so far is the CUDA source based JIT compilation model. While it is convenient it has a number of downsides:

- **High compilation times**: Majority of the JIT compilation time is spent on the CUDA C++ frontend. The kernel's translation unit contains a non-trivial amount of code from cuDF and its dependencies that are processed on every compilation request. Most of the contents of the preprocessed translation unit are redundant but still get processed anyway.
- **Correctness**: Source based JIT compilation is difficult to guarantee compilation correctness for because the code is not compiled until at runtime. 

Link-time-Optimization (LTO) JIT helps solve both problems. LTO JIT is similar to the C++ translation unit linking process.

LTO JIT allows users to define functions in separate translation units and link them at runtime.
For example, the translation unit above can be compiled and shipped as-is without defining the `transform` function.
This effectively reduces the runtime work solely to JIT-linking as no time is spent on the CUDA C++ frontend provided both the kernel and the UDF have been compiled Ahead-of-Time (AOT).
Just like CUDA Source-Based JIT, LTO JIT allows users to extend the behaviour of cuDF's kernels and implement new functionality but with the upside of faster JIT times and better AOT correctness checks.
The CUDA source code is compiled to a binary program representation called LTO IR.


### 1. Kernel Template Construction

An LTO-compatible binary operation kernel template will be defined as:

```cpp

// file: binary_op_kernel.cu

using transform_type = CUDF_UDF_TYPE;

extern "C" transform_type transform;

template<typename LhsType, typename RhsType, bool RhsIsScalar, typename OutputType>
__device__ void binary_op_kernel(column_device_view const * inputs, mutable_column_device_view const * output, size_type n){
    auto start  = detail::grid_1d::global_thread_id();
    auto stride = detail::grid_1d::grid_stride();

    for (auto i = start; i < n; i += stride) {
        transform(&output->element<OutputType>(i), inputs[0]->element<LhsTypee>(i), inputs[1]->element<RhsType>(i * RhsIsScalar));
    }
}

extern "C" __device__ void cudf_kernel_entry(column_device_view const * inputs, mutable_column_device_view const * outputs, size_type n){
    CUDF_KERNEL_INSTANCE(inputs, outputs, n);
}

```

A few things to note:
- The kernel template above is same as the CUDA source based kernel template with the addition of an external linkage `transform` function
- `CUDF_UDF_TYPE` specifies the function signature for `transform` that will match the `binary_op_kernel`



### 2. Kernel Compilation, Embedding & Registration

A compiled unit of CUDA code is called a *fragment*. 
A compiled kernel to be linked later is called a *kernel fragment* (e.g. `binary_op_kernel`)
A compiled UDF to be linked later to a kernel is called a *UDF fragment* (e.g. `transform`)

You can compile and embed an instance of the kernel above using the `add_fragment` helper function


```cmake
add_fragment(
    cudf_fragments # <- ID of cuDF's fragments
    FRAGMENT
    binary_op_kernel_i32_float_false_float # <- Fragment instance identifier
    SOURCE
    binary_op_kernel.cu # <- Location of the CUDA source code
    KERNEL_INSTANCE
    binary_op_kernel<int32_t, float, false, float> # <- Instance of the templated kernel to precompile
    UDF_TYPE
    "int(float *, int32_t, int32_t)" #<- The matching UDF type
)
```


A user can compile and embed their UDF using the same helper function.
Provided they have a UDF:

```cpp
// file: user_udf.cu
extern "C" __device__ void transform(float * out, int32_t lhs, int32_t rhs){
    *out = (lhs + rhs) * 0.5F;
}
```

The UDF above can be registered by the user into their translation unit with:

```cmake
add_fragment(
    user_fragments
    FRAGMENT
    user_udf
    SOURCE
    user_udf.cu
)
```


### 3. Kernel Execution

The pre-compiled fragments can be linked and executed at runtime using `cudf::get_lto_linked_kernel`:


```cpp
#include <cudf_fragments.hpp>
#include <user_fragments.hpp>


auto kernel_range = cudf_fragments::file_ranges[cudf_fragments::binary_op_kernel_i32_float_false_float];
std::span kernel_fragment = cudf_fragments::files.subspan(range[0], range[1]);
auto udf_range = user_fragments::file_ranges[user_fragments::user_udf];
auto udf_fragment = user_fragments::files.subspan(range[0], range[1]);
auto kernel = cudf::get_lto_linked_kernel("binary_op_kernel", {}, {kernel_fragment, udf_fragment});
column_device_view const * inputs = ...;
mutable_column_device_view const * outputs = ...;
size_type n = ...;
kernel.launch_with({grid_size}, {block_size}, 0, stream, inputs, outputs, n);
```


### 4. Hybrid JIT Model

cuDF uses a hybrid of the CUDA JIT and LTO JIT models for the generic multi-output multi-input transform kernels. 

An example cuDF transform kernel is:

```cpp
template<template<typename ... I> Inputs, template<typename ... O> Outputs>
extern "C" __device__ void transform_kernel(column_device_view const * inputs, mutable_column_device_view const * output, size_type n);
```

This kernel has M inputs and N outputs. Each input and output would span ~30 element types (`int32`, `float32`, `decimal32`, etc.). Pre-compiling for all kernel configurations will increase compilation time and bloat binary size severely.
This necessitates an hybrid approach: Some commonly used kernel fragments are pre-compiled Ahead-Of-Time while the rest are compiled and cached at JIT time.