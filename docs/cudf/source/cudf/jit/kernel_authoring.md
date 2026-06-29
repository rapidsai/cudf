

## CUDA Kernel Authoring for JIT Compilation

This document explains the mental model for how to develop CUDA kernels for JIT compilation in libcudf. It is intended for developers who are familiar with CUDA programming and want to write custom kernels that can be JIT compiled at runtime.

Kernel specialization in cuDF is of two types:
- Kernel Configuration Specialization: this determines the physical shape of the kernel, i.e: memory layout, nullability, vectorization, input types, output types, etc. It directly affects bandwidth usage, memory access patterns, and register usage of the kernel.
- Kernel UDF Specialization: this determines the logical behaviour of the kernel, i.e: the operations performed, the computation logic, etc. It directly affects the number of instructions, instruction-level parallelism, register pressure, and instruction-level optimizations of the kernel.

### 1. Kernel Template Construction

The kernel source template is a CUDA source code which is templatized to allow for kernel instantiation and specialization.

An illustrative kernel source template for a unary operation kernel that takes a single input column and produces a single output column is shown below:

```cpp

// file: my_unary_op_kernel.cu
#include <cudf/detail/kernel_instance.cuh> // runtime-specialized contains `CUDF_KERNEL_INSTANCE` macro
#include <cudf/detail/operation_udf.cuh> // runtime-specialized contains `GENERIC_TRANSFORM_OP` macro

template<typename InputType, bool input_is_scalar, typename OutputType>
__device__ void unary_op_kernel(column_device_view const * input, mutable_column_device_view const * output, size_type n){
       auto start  = detail::grid_1d::global_thread_id();
       auto stride = detail::grid_1d::grid_stride();

        for (auto i = start; i < n; i += stride) {
            GENERIC_TRANSFORM_OP(&output->element<OutputType>(i), input->element<InputType>(i * input_is_scalar));
        }
}


extern "C" __device__ void cudf_kernel_entry(column_device_view const * inputs, mutable_column_device_view const * outputs, size_type n){
    CUDF_KERNEL_INSTANCE(inputs, outputs, n);
}

```

A few things to note:

- The kernel template `unary_op_kernel` is templatized on the input and output types, allowing for instantiation and specialization of the kernel across each input and output type combination.
- The `cudf_kernel_entry` function is the scope-independent entry point for the kernel, which is used to reference the kernel entry point and allows the entry point to be referenced independently of the kernel's template parameters as it is not tied to a complex mangling scheme.
- The `CUDF_KERNEL_INSTANCE` macro is used to instantiate/specialize the kernel for each input and output type combination and it is not known or resolved until runtime, this represents the kernel configuration specialization.
- The `GENERIC_TRANSFORM_OP` macro is used to instantiate/specialize the kernel for each UDF and it is not known or resolved until runtime, representing the kernel UDF specialization.


### 2. Source Embedding & Registration

The kernel source template above is embedded into the libcudf binary as a compressed string literal. This allows for the kernel source template to be specialized at runtime and compiled into a kernel for execution on the GPU. The kernel source templates are registered and embedded into libcudf through the CMake function `embed_includes` provided by librtcx.

e.g.

```cmake

embed_includes(
  cudf_cuda_embed # ID of target to embed the source template into
  SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_unary_op_kernel/jit # directory containing the kernel source template
  DEST_DIRECTORY cudf/cpp/src/sample_unary_op_kernel/jit # directory on the host where the source_directory will be installed to
  INCLUDE_DIRECTORIES cudf/cpp/src # include directory to be added for the target
)

```


### 3. Kernel Runtime Reflection & Specialization

At runtime, the kernel is instantiated and specialized based on runtime reflection of runtime arguments. This means runtime values/arguments are used to determine the kernel's type and configuration.
This primarily involves mapping runtime input and output types to the kernel template's type parameters, which allows for the kernel to be specialized for the specific input and output types at runtime.

e.g.

<!-- construct mermaid diagram to better illustrate this -->

```cpp
// reflection based on runtime parameters
std::string reflect_unary_operation_kernel(data_type input_type /* = type_id::INT32 */, bool input_is_scalar /* = false */, data_type output_type /* = type_id::FLOAT32 */); /* -> ::unary_op_kernel<int32_t, false, float> */

// kernel configuration specialization (substituted during call to `cudf::get_udf_kernel`)
#define CUDF_KERNEL_INSTANCE ::unary_op_kernel<int32_t, false, float>

```

Note: The mapping to the kernel instance can be more sophisticated and include policies like: optimal kernel configurations, memory layouts, vectorization policies, etc. which can be determined at runtime based on the input and output types and their properties.


### 4. UDF Specialization

At runtime, the User-Defined Function (UDF) is resolved to CUDA code and specializes the `GENERIC_TRANSFORM_OP` macro in the kernel template. This allows for the kernel's generated binary code to be specialized for the specific UDF at runtime, allowing for register & instruction optimizations and inlining of the UDF into the kernel's generated binary code.

e.g.

```cpp

// runtime UDF
__device__ void as_float(float * output, int input){
    *output = static_cast<float>(input);
}

// kernel UDF specialization (substituted during call to `cudf::get_udf_kernel`)
#define GENERIC_TRANSFORM_OP(output, input) ::as_float(output, input)

```



### 5. Kernel Compilation & Caching

Given that the kernel UDF and the kernel templates have been resolved at runtime, the kernel can be compiled for execution on the GPU.
The kernel compilation process involves a complex set of steps that authors need not worry about, this generally goes through the `cudf::get_udf_kernel` function, which is responsible for generating the specialized CUDA kernel source code, compiling the kernel, caching it, and returning a handle (`cudf::kernel`) to the compiled kernel.



### 6. Kernel Execution

The compiled kernel can be launched on the GPU using the `cudf::kernel::launch` function with the corresponding kernel handle and the kernel's arguments.
Given that the kernel uses a static entry point, it is recommended that the kernel's arguments are type-erased and non-dependent on the kernel's template parameters. This allows for the kernel to be launched independently of the kernel's template parameters.

```cpp
auto kernel_instance = "::unary_op_kernel<int32_t, false, float>";
auto udf = "__device__ void as_float(float * output, int input){ *output = static_cast<float>(input); }";
auto kernel = cudf::get_udf_kernel("my_unary_op_kernel.cu", kernel_instance, udf);
column_device_view const * input = ...;
mutable_column_device_view const * output = ...;
size_type n = ...;
kernel.launch_with(input, output, n);
```

<!-- Generate NVIDIA-style diagram to illustrate the entire flow -->

NOTE: As of 26.06 PTX UDFs are supported by converting them to CUDA C++ code by using the `asm` CUDA directive. This still pays the full CUDA C++ frontend compilation cost.
