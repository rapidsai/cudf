# JIT LTO (Just-In-Time Link-Time Optimization) Guide

## Background

### What is JIT LTO?

[JIT LTO (Just-In-Time Link-Time Optimization)](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/) is a CUDA compilation strategy that enables dynamic kernel compilation and linking at runtime. Instead of pre-compiling all possible kernel variants (which would result in an explosion of binary size), JIT LTO compiles kernel **fragments** separately and links them together on-demand when a specific kernel configuration is needed.

### Fragment Terminology

A **fragment** is a self-contained, compilable unit of CUDA code that can be linked with other fragments to form a complete kernel. In the JIT LTO system:

- **Entrypoint Fragment**: The main kernel function that serves as the entry point. This is always the `__global__` kernel function.
- **Device Function Fragments**: Separate fragments containing device functions (e.g., distance computations, filters, post-processing) that are called by the entrypoint kernel.
- **Fragment Key**: A unique identifier for a fragment, typically constructed from template parameters and configuration values.
- **Fatbin**: The compiled binary representation of a fragment, embedded in the executable.

The key advantage is that device functions can be compiled independently and reused across multiple kernel entrypoints, reducing compilation time and binary size.

### How It Works

1. **Build Time**: Fragments are compiled into fatbins and embedded in the executable.
2. **Runtime**: When a kernel needs to be launched:
   - The planner identifies which fragments are needed based on the configuration
   - Fragments are loaded from the embedded fatbins
   - Nvjitlink (Link-Time Optimization) links the fragments together
   - The linked kernel is cached and launched

## Walkthrough Example

Let's walk through creating a JIT LTO kernel system for a search kernel with templated device functions.

### Step 1: Define the Kernel and Device Functions

We start with a kernel that has templated device functions that we want to separate into fragments:

**`search_kernel.cuh`**:

```cpp
#pragma once

#include <cuda_runtime.h>

namespace example::detail {

// Device function for distance computation
template <typename T>
__device__ float compute_distance_euclidean(T a, T b) {
    T diff = a - b;
    return diff * diff;
}

template <typename T>
__device__ float compute_distance_inner_product(T a, T b) {
    return -a * b;  // Negative for max inner product search
}

// Device function for filtering
template <typename IdxT>
__device__ bool apply_filter_none(uint32_t query_id, IdxT node_id, void* filter_data) {
    return true;
}

template <typename IdxT>
__device__ bool apply_filter_bitset(uint32_t query_id, IdxT node_id, void* filter_data) {
    // Simplified - actual implementation would check bitset
    return true;
}

// Main kernel - will use generic extern device functions
template <typename T, typename OutT, typename IdxT, bool UseOptimizedPath, int Veclen>
__device__ void search_kernel_impl(
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,  // Output distance type
    uint32_t num_queries,
    uint32_t dataset_size,
    void* filter_data) {

    uint32_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;

    OutT best_dist = std::numeric_limits<OutT>::max();
    IdxT best_idx = 0;

    for (IdxT i = 0; i < dataset_size; ++i) {
        // Call generic extern device functions (implementations linked from fragments)
        if (!apply_filter<IdxT>(query_id, i, filter_data)) continue;

        OutT dist = static_cast<OutT>(compute_distance<T>(queries[query_id], dataset[i]));

        // Use optimized path if enabled
        if constexpr (UseOptimizedPath) {
            // Optimized implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        } else {
            // Standard implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
    }

    results[query_id] = best_idx;
    distances[query_id] = best_dist;
}

} // namespace example::detail
```

### Step 2: Create Device Function Fragments

We'll create separate header files for each device function variant. Each implements the generic function signature that the kernel expects:

**`compute_distance_euclidean.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic compute_distance function for euclidean distance
template <typename T>
__device__ float compute_distance(T a, T b) {
    T diff = a - b;
    return diff * diff;
}

} // namespace example::detail
```

**`compute_distance_inner_product.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic compute_distance function for inner product
template <typename T>
__device__ float compute_distance(T a, T b) {
    return -a * b;  // Negative for max inner product search
}

} // namespace example::detail
```

**`filter_none.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic apply_filter function for no filtering
template <typename IdxT>
__device__ bool apply_filter(uint32_t query_id, IdxT node_id, void* filter_data) {
    return true;
}

} // namespace example::detail
```

**`filter_bitset.cuh`**:

```cpp
#pragma once

namespace example::detail {

// Implements the generic apply_filter function for bitset filtering
template <typename IdxT>
__device__ bool apply_filter(uint32_t query_id, IdxT node_id, void* filter_data) {
    // Actual bitset implementation
    return true;
}

} // namespace example::detail
```

### Step 3: Create JSON Matrix Files

JSON matrix files define all the parameter combinations that need to be compiled. The build system uses these to generate `.cu` files from `.cu.in` templates.

**How JSON Cross-Product Works**:
- The build system computes a modified **Cartesian product** (cross-product) of all parameter combinations.
- **Leaf nodes** are the actual values. These can be strings, numbers, booleans, or `null`, but only strings should be used, even for numbers, for example ``"1"``.
- Related values can be grouped together in a dictionary consisting of single values. Any dictionary key in such a dictionary's ancestry will not be used in the final product, and should be prefixed with `_` to indicate that it is used only for grouping.
- Keys containing only leaf nodes will be used in the final product, and should not be prefixed with `_`.
- The matrix product algorithm will automatically warn if the proper naming convention (`_` prefix or not) is not followed.
- Each group expands to create multiple combinations, and all groups are cross-multiplied.

For example, if you have:
```json
{
  "_data_type": [{"data_type": "float"}, {"data_type": "half"}],
  "_index": [{"idx_type": "uint32_t"}, {"idx_type": "int64_t"}],
  "capacity": ["1", "2"]
}
```

This generates 2 × 2 × 2 = 8 combinations:
- `{data_type: "float", idx_type: "uint32_t", capacity: "1"}`
- `{data_type: "float", idx_type: "uint32_t", capacity: "2"}`
- `{data_type: "float", idx_type: "int64_t", capacity: "1"}`
- ... and so on

When a group contains nested arrays (like `veclen: ["1", "4"]`), those are also expanded within that group before the cross-product is computed.

#### `compute_distance_matrix.json`

```json
{
  "_distance_type": [
    {
      "distance_name": "euclidean",
      "header_file": "example/jit_lto_kernels/compute_distance_euclidean.cuh"
    },
    {
      "distance_name": "inner_product",
      "header_file": "example/jit_lto_kernels/compute_distance_inner_product.cuh"
    }
  ],
  "_data_type": [
    {
      "data_type": "float",
      "type_abbrev": "f"
    },
    {
      "data_type": "__half",
      "type_abbrev": "h"
    }
  ]
}
```

#### `filter_matrix.json`

```json
{
  "filter_name": [
    "filter_none",
    "filter_bitset"
  ],
  "_index": [
    {
      "idx_type": "uint32_t",
      "idx_abbrev": "ui"
    },
    {
      "idx_type": "int64_t",
      "idx_abbrev": "l"
    }
  ]
}
```

#### `search_kernel_matrix.json`

This example demonstrates conditional combinations: `OutT` can be `float` or `double` when `T` is `float`, but only `float` when `T` is `__half`.

```json
{
  "_data_type": [
    {
      "data_type": "float",
      "type_abbrev": "f",
      "_output_type": [
        {
          "out_type": "float",
          "out_abbrev": "f"
        },
        {
          "out_type": "double",
          "out_abbrev": "d"
        }
      ]
    },
    {
      "data_type": "__half",
      "type_abbrev": "h",
      "_output_type": [
        {
          "out_type": "float",
          "out_abbrev": "f"
        }
      ]
    }
  ],
  "_index": [
    {
      "idx_type": "uint32_t",
      "idx_abbrev": "ui"
    },
    {
      "idx_type": "int64_t",
      "idx_abbrev": "l"
    }
  ],
  "_optimized": [
    {
      "optimized_name": "optimized",
      "optimized_value": "true",
      "veclen": ["1", "4"]
    },
    {
      "optimized_name": "standard",
      "optimized_value": "false",
      "veclen": ["8", "16"]
    }
  ]
}
```

This generates 24 combinations (3 data/output type combinations × 2 index types × 4 optimized/veclen combinations):
- `float` + `float` + `uint32_t` + `optimized` + `veclen=1`
- `float` + `float` + `uint32_t` + `optimized` + `veclen=4`
- `float` + `float` + `uint32_t` + `standard` + `veclen=8`
- `float` + `float` + `uint32_t` + `standard` + `veclen=16`
- `float` + `double` + `uint32_t` + `optimized` + `veclen=1`
- `float` + `double` + `uint32_t` + `optimized` + `veclen=4`
- `float` + `double` + `uint32_t` + `standard` + `veclen=8`
- `float` + `double` + `uint32_t` + `standard` + `veclen=16`
- `__half` + `float` + `uint32_t` + `optimized` + `veclen=1`
- `__half` + `float` + `uint32_t` + `optimized` + `veclen=4`
- `__half` + `float` + `uint32_t` + `standard` + `veclen=8`
- `__half` + `float` + `uint32_t` + `standard` + `veclen=16`
- ... and the same with `int64_t` (total: 24 combinations)

### Step 4: Create `.cu.in` Template Files

The `.cu.in` files are templates that get instantiated for each combination in the JSON matrix. They contain explicit template instantiations.

#### `compute_distance_kernel.cu.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "@header_file@"

namespace example::detail {

// Instantiate the generic compute_distance device function template
// The specific implementation (euclidean or inner_product) comes from the header
template __device__ float compute_distance<@data_type@>(@data_type@, @data_type@);

} // namespace example::detail
```

#### `filter_kernel.cu.in`

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example/jit_lto_kernels/@filter_name@.cuh"

namespace example::detail {

// Instantiate the generic apply_filter device function template
// The specific implementation (filter_none or filter_bitset) comes from the header
template __device__ bool apply_filter<@idx_type@>(uint32_t, @idx_type@, void*);

} // namespace example::detail
```

#### Update `search_kernel.cuh` with Extern Declarations

The kernel header needs to declare generic extern device functions so the kernel code can call them. The specific implementations will be linked from fragments at runtime:

**`search_kernel.cuh`**:

```cpp
#pragma once

#include <cuda_runtime.h>

namespace example::detail {

// Forward declare generic extern device functions that will be linked from fragments
// The specific implementations (euclidean, inner_product, etc.) are resolved at link time
template <typename T>
extern __device__ float compute_distance(T, T);

template <typename IdxT>
extern __device__ bool apply_filter(uint32_t, IdxT, void*);

// Main kernel - uses generic extern device functions
template <typename T, typename OutT, typename IdxT, bool UseOptimizedPath, int Veclen>
__device__ void search_kernel_impl(
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,  // Output distance type
    uint32_t num_queries,
    uint32_t dataset_size,
    void* filter_data) {

    uint32_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;

    OutT best_dist = std::numeric_limits<OutT>::max();
    IdxT best_idx = 0;

    for (IdxT i = 0; i < dataset_size; ++i) {
        // Call generic extern device functions (specific implementations linked from fragments)
        if (!apply_filter<IdxT>(query_id, i, filter_data)) continue;

        OutT dist = static_cast<OutT>(compute_distance<T>(queries[query_id], dataset[i]));

        // Use optimized path if enabled
        if constexpr (UseOptimizedPath) {
            // Optimized implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        } else {
            // Standard implementation
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
    }

    results[query_id] = best_idx;
    distances[query_id] = best_dist;
}

} // namespace example::detail
```

#### `search_kernel.cu.in`

The `.cu.in` file only contains the explicit template instantiation:

```text
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example/jit_lto_kernels/search_kernel.cuh"

namespace example::detail {

// Instantiate the kernel template
extern "C" __global__ void search_kernel(
    const @data_type@* dataset, const @data_type@* queries, @idx_type@* results, @out_type@* distances,
    uint32_t num_queries, uint32_t dataset_size, void* filter_data)
{
  search_kernel_impl<@data_type@, @out_type@, @idx_type@, @optimized_value@, @veclen@>(
    dataset,
    queries,
    results,
    distances,
    num_queries,
    dataset_size,
    filter_data);
}

} // namespace example::detail
```

**Note**: The kernel uses generic function templates (`compute_distance<T>` and `apply_filter<IdxT>`) that are resolved at link time. The specific implementations (euclidean vs inner_product, filter_none vs filter_bitset) are provided by the fragments that get linked together.

### Step 5: Create Fragment Tags for Embedding

Fragment tags register the compiled fatbins so they can be loaded at runtime. They are used to help the linker find and include the relevant fatbins at build time. When calling `generate_jit_lto_kernels()`, we pass a `FRAGMENT_TAG_FORMAT` argument, which constructs the tag type from the given placeholders, and a `FRAGMENT_TAG_HEADER_FILES` argument, which specifies one or more header files that the fragment tags come from. The JIT+LTO system will then automatically generate and compile a `.cpp` file that registers the fragment using the provided tag.

**Important**: When requesting fragments from the `AlgorithmPlanner`, we use **tags** (like `tag_f`, `tag_h`) instead of real types (like `float`, `__half`) in the `add_static_fragment` template parameters. This avoids including heavy headers that define the actual types, significantly improving compilation times. The tags are lightweight empty structs that serve only as compile-time identifiers.

**`registration_tags.hpp`**

```cpp
#pragma once

struct tag_h{};
struct tag_f{};
struct tag_d{};
struct tag_ui{};
struct tag_l{};

struct tag_metric_euclidean {};
struct tag_metric_inner_product {};

struct tag_filter_none {};
struct tag_filter_bitset {};

template <typename DataTag, typename OutTag, typename IdxTag, bool Optimized, int Veclen>
struct fragment_tag_search {};

template <typename DistanceTag, typename DataTag>
struct fragment_tag_compute_distance {};

template <typename FilterTag, typename IndexTag>
struct fragment_tag_filter {};
```

### Step 6: Create the Planner

The planner is responsible for:
1. Identifying which fragments are needed for a given configuration
2. Building a unique key for the fragment combination
3. Requesting the fragments from the fragment database
4. Linking them together to create a launchable kernel

**CRITICAL**: The fragment keys constructed in the planner methods must match **EXACTLY** with the keys used in the corresponding `FRAGMENT_TAG_FORMAT` argument. Any mismatch will result in runtime linking failures.

**`search_planner.hpp`**:

```cpp
#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <string>

struct SearchPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  SearchPlanner()
    : AlgorithmPlanner("search_kernel", launcher_jit_cache)
  {
  }

  template <typename DataTag, typename OutTag, typename IdxTag, bool Optimized, int Veclen>
  void add_search_function()
  {
    add_static_fragment<fragment_tag_search<DataTag, OutTag, IdxTag, Optimized, Veclen>>();
  }

  template <typename DistanceTag, typename DataTag>
  void add_compute_distance_function()
  {
    add_static_fragment<fragment_tag_compute_distance<DistanceTag, DataTag>>();
  }

  template <typename FilterTag, typename IndexTag>
  void add_filter_function()
  {
    add_static_fragment<fragment_tag_filter<FilterTag, IndexTag>>();
  }
};
```

### Step 7: Integrate with Code Path

Now we integrate the planner into the actual search function:

**`search_jit.cuh`**:

```cpp
#pragma once

#include "search_planner.hpp"
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <raft/core/device_resources.hpp>

namespace example::detail {

// Type tag helpers
template <typename T>
constexpr auto get_data_type_tag() {
  if constexpr (std::is_same_v<T, float>) return tag_f{};
  if constexpr (std::is_same_v<T, __half>) return tag_h{};
}

template <typename IdxT>
constexpr auto get_idx_type_tag() {
  if constexpr (std::is_same_v<IdxT, uint32_t>) return tag_ui{};
  if constexpr (std::is_same_v<IdxT, int64_t>) return tag_l{};
}

template <typename OutType>
constexpr auto get_out_type_tag() {
  if constexpr (std::is_same_v<OutType, float>) return tag_f{};
  if constexpr (std::is_same_v<OutType, double>) return tag_d{};
}

template <DistanceType Metric>
constexpr auto get_metric_tag() {
  if constexpr (Metric == DistanceType::Euclidean) return tag_metric_euclidean{};
  if constexpr (Metric == DistanceType::InnerProduct) return tag_metric_inner_product{};
}

template <FilterType Filter>
constexpr auto get_filter_tag() {
  if constexpr (Filter == FilterType::None) return tag_filter_none{};
  if constexpr (Filter == FilterType::Bitset) return tag_filter_bitset{};
}

template <typename T, typename OutT, typename IdxT, DistanceType Metric, FilterType Filter, bool Optimized, int Veclen>
void search_jit(
    raft::device_resources const& handle,
    const T* dataset,
    const T* queries,
    IdxT* results,
    OutT* distances,
    uint32_t num_queries,
    uint32_t dataset_size,
    void* filter_data = nullptr) {

  using data_tag = decltype(get_data_type_tag<T>());
  using idx_tag = decltype(get_idx_type_tag<IdxT>());
  using out_tag = decltype(get_out_type_tag<OutT>());
  using metric_tag = decltype(get_metric_tag<Metric>());
  using filter_tag = decltype(get_filter_tag<Filter>());

  // Create planner with type tags and boolean parameter
  // Note: The boolean is appended to the fragment key since make_fragment_key
  // cannot handle non-type template parameters
  SearchPlanner planner;

  // Add required device function fragments
  planner.add_search_function<data_tag, out_tag, idx_tag, Optimized, Veclen>();
  planner.add_compute_distance_device_function<metric_tag, data_tag>();
  planner.add_filter_device_function<filter_tag, idx_tag>();

  // Get the launcher (this will build/link fragments if needed)
  auto launcher = planner.get_launcher();

  // Launch configuration
  dim3 block(256);
  dim3 grid((num_queries + block.x - 1) / block.x);

  // Launch the kernel - arguments are passed directly
  launcher->dispatch(
      raft::resource::get_cuda_stream(handle),
      grid,
      block,
      0,  // shared memory size
      dataset,
      queries,
      results,
      distances,
      num_queries,
      dataset_size,
      filter_data);
}

} // namespace example::detail
```

## Key Concepts

### Fragment Tags

Fragment tags uniquely identify fragments. They're simple lightweight types that are passed as the
sole template parameter to `StaticFatbinFragmentEntry`:

```cpp
template <typename OutT>
struct fragment_tag_get_score {};
```

Fragment tags may themselves take template parameters in order to uniquely identify them. Typically, one fragment tag template will correspond to a single function, and a fragment tag template specialization will correspond to a function specialization.

When a fatbin is compiled and embedded in C++ code, a translation unit specializes `StaticFatbinFragmentEntry`
to specify its `data` and `length` static fields:

```cpp
using _FragmentEntry = StaticFatbinFragmentEntry<fragment_tag_get_score<uint32_t>>;

template <>
const uint8_t* const _FragmentEntry::data = embedded_fatbin;

template <>
const size_t _FragmentEntry::length = sizeof(embedded_fatbin);
```

Then, an `AlgorithmPlanner` can call `add_static_fragment()` with the fragment tag (NOT the `StaticFatbinFragmentEntry`
specialization) as the sole template parameter:

```cpp
template <typename OutTag>
void add_get_score_function()
{
  add_static_fragment<fragment_tag_get_score<OutTag>>();
}
```

At build time, the linker takes care of finding and including the static fragments that have been specified by the
algorithm planner.

### Registration Tags

Registration tags are type-safe identifiers used to organize fragments. They're typically empty structs:

```cpp
struct tag_f {};  // float
struct tag_h {};  // half
struct tag_ui {}; // uint32_t
struct tag_l {};  // int64_t
```

These tags are used in `registerAlgorithm<>()` to create a hierarchical organization of fragments.

### AlgorithmLauncher

The `AlgorithmLauncher` is the runtime handle for a linked kernel. It:
- Holds a `cudaKernel_t` handle to the linked kernel
- Provides `call()` and `call_cooperative()` methods to launch the kernel
- Manages the lifetime of the `cudaLibrary_t` that contains the kernel

## Best Practices

1. **Minimize Includes**: JIT LTO fragments should have minimal includes, especially avoiding host-side headers. Extract device-only code into separate headers.

2. **Fragment Granularity**: Balance between too many small fragments (overhead) and too few large fragments (less reuse). Device functions that are reused across multiple kernels are good candidates for separate fragments.

3. **Naming Consistency**: Ensure fragment tags match exactly between registration and lookup. Use helper functions to construct tags consistently.

4. **Type Safety**: Use registration tags to provide compile-time type safety and avoid runtime string mismatches.

5. **Caching**: Each planner type should hold a static `LauncherJitCache` and pass it to `AlgorithmPlanner`; `get_launcher()` then reuses linked kernels for the same fragment key within that cache.

## Example: IVF Flat

IVF Flat uses JIT LTO with:
- **Metric fragments**: Euclidean and inner product distance computations (16 fatbins)
- **Post-lambda fragments**: Identity, sqrt, and compose post-processing (3 fatbins)
- **Interleaved scan fragments**: Main search kernel with various configurations (320 fatbins)
- **Filter fragments**: None and bitset filters (2 fatbins)

**Total: 341 fatbins** that can be combined into many more kernel variants at runtime.

### Step 8: Integrate with CMake Build System

To integrate JIT LTO kernels into the CMake build system, add calls to `generate_jit_lto_kernels()` in your main `CMakeLists.txt` file (typically in `cpp/CMakeLists.txt`).

The `generate_jit_lto_kernels()` function (defined in `cmake/modules/generate_jit_lto_kernels.cmake`) takes:
- `NAME_FORMAT`: Format string for generated kernel names (using `@variable@` syntax)
- `MATRIX_JSON_FILE`: Path to the JSON matrix file
- `KERNEL_INPUT_FILE`: Path to the `.cu.in` template
- `FRAGMENT_TAG_FORMAT`: Format string for fragment tag type (using `@variable@` syntax)
- `FRAGMENT_TAG_HEADER_FILES`: List of header files that provide the fragment tag types (can be enclosed in `<`/`>` or `"`/`"`, automatically enclosed in quotes if quotes and brackets are not provided)
- `OUTPUT_DIRECTORY`: Where generated files are placed
- `KERNEL_LINK_LIBRARIES`: Interface library with compilation settings

Call `generate_jit_lto_kernels()` once for each fragment type (compute_distance, filter, search_kernel, etc.). The function reads the JSON matrix, computes the cross-product of all combinations, generates `.cu` and `.cpp` files from the templates, compiles them into fatbins, and returns a list of generated source files that should be added to your JIT LTO library target.

See the CUVS `cpp/CMakeLists.txt` file for a complete example of how to set up the interface library, call `generate_jit_lto_kernels()` for each fragment type, and create the final library target.

## Summary

JIT LTO enables:
- **Reduced binary size**: Compile fragments once, combine many ways
- **Faster compilation**: Fragments compile independently
- **Runtime flexibility**: Link fragments on-demand based on configuration
- **Code reuse**: Device function fragments shared across kernels

The process involves:
1. Separating device functions into fragment headers
2. Creating JSON matrices defining parameter combinations
3. Creating `.cu.in` templates for explicit instantiations
4. Creating fragment tag types for fatbin registration
5. Creating a planner to manage fragment dependencies
6. Integrating the planner into the code path to launch kernels
7. **Adding CMake integration** to generate and compile all fragment variants

## Fragment Architecture

JIT LTO kernels are split into _fragments_, which are fatbins containing individual pieces of code that can be strung together
rather than instantiating the whole kernel at once. Each fragment only needs to be multiplied out over the dimensions (template
parameters) that the fragment itself contains rather than the kernel as a whole. At runtime, these fragments are combined together
by nvjitlink into the final program.

In JIT LTO, there are two kinds of code: _algorithms_ and _adapters_. Algorithms are, roughly speaking, code that actually "does
stuff" - searching, sorting, even as simple as initializing variables. Adapters don't do anything by themselves, but are merely
thin wrappers around algorithms that exist only for reducing the number of template parameters that the caller needs to know about.
It should generally be assumed that algorithm code is expensive to multiply over a matrix, and thus such multiplication should be
minimized, while adapter code is cheap to multiply.

An algorithm function is a function that contains real code for the algorithm, and an adapter function merely calls an algorithm
function with more template parameters than the adapter function itself has. An algorithm file contains algorithm code, and an
adapter file contains adapter code.

Here is an example of an algorithm file that contains an algorithm function:

```c++
template <typename T, T Divisor>
__device__ bool is_divisible_impl(T value)
{
  return value % Divisor == 0;
}
```

Here is an example of an adapter file that contains an adapter function:

```
#include "device_functions.cuh"  // is_divisible
#include "is_divisible_impl.cuh" // is_divisible_impl

namespace {

using data_t             = @data_type@;
constexpr data_t divisor = @divisor@;

}  // namespace

template <>
__device__ bool is_divisible<data_t>(data_t value)
{
  return is_divisible_impl<data_t, divisor>(value);
}
```

This is the most common pattern that you will see in cuVS's JIT LTO code. Note that any code that calls `is_divisible()` does not
need to know the value of `Divisor`, which allows the caller to be multiplied over fewer dimensions, thus reducing the amount of code
generated.

Note that in the above adapter file, `@data_type@` and `@divisor@` are build-time substitutions performed by CMake. These
substitutions will be filled in with values from the matrix product. Note that they are all grouped together in a single `namespace`,
making it easy to find all substitutions. This should be preferred to sprinkling the substitutions throughout the code.

Here is an example with two algorithm files:

```c++
// greater_than_impl.cuh
#include "device_impl_functions.cuh" // filter

template <typename T, T Comparand>
__device__ bool filter(T value)
{
  return value > Comparand;
}
```

```c++
// less_than_impl.cuh
#include "device_impl_functions.cuh" // filter

template <typename T, T Comparand>
__device__ bool filter(T value)
{
  return value < Comparand;
}
```

And here is the accompanying adapter file:

```
#include "@op_name@_impl.cuh" // filter

namespace {

using data_t = @data_type@;

}

template __device__ bool filter<data_t>(data_t value);
```

This is another common pattern that you will see in cuVS JIT LTO. Note that the adapter file does not contain any adapter functions,
but merely instantiates a different algorithm function based on which algorithm file is included based on the CMake substitution.

When a piece of algorithm code is used in multiple kernels, it should be split into its own shared fragment. At this point, it
becomes important to also distinguish algorithm fragments and adapter fragments. An algorithm fragment contains an algorithm function
that exposes all of the relevant template parameters, and this fragment is shared between multiple kernels. An adapter fragment
is specific to a kernel. If a kernel wishes to invoke the same shared algorithm multiple times in the same run with
different template parameters, it can employ multiple adapter fragments to accomplish this. Consider the following header file:

```c++
// filter.cuh

template <typename T, T Comparand>
__device__ bool filter_less_than(T value);

template <typename T, T Comparand>
__device__ bool filter_greater_than(T value);
```

And the following adapter files:

```
#include "device_functions.cuh" // filter_first_pass
#include "filter.cuh"           // filter

namespace {

using data_t               = @data_type@;
constexpr data_t comparand = @comparand@;

}

template <>
__device__ bool filter_first_pass<data_t>(data_t value)
{
  return filter_@op_name@<data_t, comparand>(value);
}
```

```
#include "device_functions.cuh" // filter_second_pass
#include "filter.cuh"           // filter

namespace {

using data_t               = @data_type@;
constexpr data_t comparand = @comparand@;
}

template <>
__device__ bool filter_second_pass<data_t>(data_t value)
{
  return filter_@op_name@<data_t, comparand>(value);
}
```

And the following algorithm file:

```c++
#include "device_functions.cuh" // filter_first_pass, filter_second_pass

template <typename T>
__device__ bool filter_all_passes(T value)
{
  return filter_first_pass<T>(value) && filter_second_pass<T>(value);
}
```

Note that `filter_first_pass` and `filter_second_pass` both invoke one of the `filter` functions, but which one they invoke is
decided independently for each. Also note that neither of the adapter fragments contains the underlying algorithm code, but rather
links against the corresponding shared algorithm fragments.

The key to minimizing code generation is to minimize the number of dimensions that any given fragment needs to be multiplied out
over. If a section of algorithm code uses lots of template parameters, try to separate out sections that use only a subset of
these parameters, put them into their own fragment, and remove the corresponding template parameters from the caller. Make judicious
use of adapter code to accomplish this. An adapter function should only have the template parameters that appear in its signature,
whereas an algorithm function should have all of the template parameters that appear in its signature or its implementation.

Unoptimized algorithm:

```c++
#include "filter_less_than.cuh"

template <typename T, T Comparand>
__device__ size_t find_first(T* values, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    if (filter_less_than_impl<T, Comparand>(values[i])) {
      return i;
    }
  }

  // Could not find any
  return count;
}
```

Note that the algorithm includes the `Comparand` template parameter, which means the entire algorithm has to be multiplied out over
all the possible values of this parameter.

Optimized algorithm:

```c++
#include "device_functions.cuh"

template <typename T>
__device__ size_t find_first(T* values, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    if (filter_less_than<T>(values[i])) {
      return i;
    }
  }

  // Could not find any
  return count;
}
```

We are now using an adapter function (possibly inside an adapter fragment) called `filter_less_than` to invoke
`filter_less_than_impl` (which may be inside a shared algorithm fragment). This allows us to hide the `Comparand` parameter
from `find_first`, which means we no longer need to multiply the entire algorithm over all possible values of `Comparand`, only the
`filter_less_than` adapter and algorithm.