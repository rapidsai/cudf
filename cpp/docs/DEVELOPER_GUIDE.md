# libcudf C++ Developer Guide

This document serves as a guide for contributors to libcudf C++ code. Developers should also refer 
to these additional files for further documentation of libcudf best practices.

* [Documentation Guide](DOCUMENTATION.md) for guidelines on documenting libcudf code.
* [Testing Guide](TESTING.md) for guidelines on writing unit tests.
* [Benchmarking Guide](TODO) for guidelines on writing unit benchmarks.

# Overview

libcudf is a C++ library that provides GPU-accelerated data-parallel algorithms for processing 
column-oriented tabular data. libcudf provides algorithms including slicing, filtering, sorting, 
various types of aggregations, and database-type operations such as grouping and joins. libcudf
serves a number of clients via multiple language interfaces, including Python and Java. Users may
also use libcudf directly from C++ code.

## Lexicon

This section defines terminology used within libcudf

### Column

A column is an array of data of a single type. Along with Tables, columns are the fundamental data 
structures used in libcudf. Most libcudf algorithms operate on columns. Columns may have a validity
mask representing whether each element is valid or null (invalid). Columns of nested types are 
supported, meaning that a column may have child columns. A column is the C++ equivalent to a cuDF
Python [series](https://docs.rapids.ai/api/cudf/stable/api.html#series)

### Element

An individual data item within a column. Also known as a row.

### Scalar

A type representing a single element of a data type.

### Table

A table is a collection of columns with equal number of elements. A table is the C++ equivalent to 
a cuDF Python [data frame](https://docs.rapids.ai/api/cudf/stable/api.html#dataframe).

### View

A view is a non-owning object that provides zero-copy access (possibly with slicing or offsets) data 
owned by another object. Examples are column views and table views.

# Directory Structure and File Naming

External/public libcudf APIs are grouped based on functionality into an appropriately titled 
header file  in `cudf/cpp/include/cudf/`. For example, `cudf/cpp/include/cudf/copying.hpp` 
contains the APIs for functions related to copying from one column to another. Note the  `.hpp` 
file extension used to indicate a C++ header file.

Header files should use the `#pragma once` include guard. 

The naming of external API headers should be consistent with the name of the folder that contains 
the source files that implement the API. For example, the implementation of the APIs found in
`cudf/cpp/include/cudf/copying.hpp` are located in `cudf/src/copying`. Likewise, the unit tests for 
the APIs reside in `cudf/tests/copying/`.

Internal API headers containing `detail` namespace definitions that are used across translation 
units inside libcudf should be placed in `include/cudf/detail`.

## File extensions

- `.hpp` : C++ header files
- `.cpp` : C++ source files
- `.cu`  : CUDA C++ source files
- `.cuh` : Headers containing CUDA device code

Only use `.cu` and `.cuh` if necessary. A good indicator is the inclusion of `__device__` and other
symbols that are only recognized by `nvcc`. Another indicator is Thrust algorithm APIs with a device
execution policy (always `rmm::exec_policy` in libcudf).

## Code and Documentation Style and Formatting

libcudf code uses [snake_case](https://en.wikipedia.org/wiki/Snake_case) for all names except in a 
few cases: template parameters, unit tests and test case names may use Pascal case, aka 
[UpperCamelCase](https://en.wikipedia.org/wiki/Camel_case). We do not use [Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), except sometimes when naming device data variables and their corresponding
host copies. Private member variables are typically prefixed with an underscore.

```c++
template <typename IteratorType>
void algorithm_function(int x, rmm::cuda_stream_view s, rmm::device_memory_resource* mr) 
{
  ...
}

class utility_class 
{
  ...
 private:
  int _rating{};
  std::unique_ptr<cudf::column> _column{};
}

TYPED_TEST_CASE(RepeatTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(RepeatTypedTestFixture, RepeatScalarCount)
{
  ...
}
```

C++ formatting is enforced using `clang-format`. You should configure `clang-format` on your 
machine to use the `cudf/cpp/.clang-format` configuration file, and run `clang-format` on all 
changed code before committing it. The easiest way to do this is to configure your editor to 
"format on save".

Aspects of code style not discussed in this document and not automatically enforceable are typically
caught during code review, or not enforced.

### C++ Guidelines

In general, we recommend following 
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). We also 
recommend watching Sean Parent's [C++ Seasoning talk](https://www.youtube.com/watch?v=W2tWOdzgXHA), 
and we try to follow his rules: "No raw loops. No raw pointers. No raw synchronization primitives." 

 * Prefer algorithms from STL and Thrust to raw loops.
 * Prefer libcudf and RMM [owning data structures and views](libcudf-data-structures) to raw pointers
   and raw memory allocation.
 * libcudf doesn't have a lot of CPU-thread concurrency, but there is some. And currently libcudf
   does use raw synchronization primitives. So we should revisit Parent's third rule and improve 
   here.

Documentation is discussed in the [Documentation Guide](DOCUMENTATION.md).

### Includes

The following guidelines apply to organizing `#include` lines.

 * Group includes by library (e.g. cuDF, RMM, Thrust, STL). `clang-format` will respect the 
   groupings and sort the individual includes within a group lexicographically.
 * Separate groups by a blank line.
 * Order the groups from "nearest" to "farthest". In other words, local includes, then includes 
   from other RAPIDS libraries, then includes from related libraries, like `<thrust/...>`, then 
   includes from dependencies installed with cuDF, and then standard headers (for example `<string>`, 
   `<iostream>`).
 * Use <> instead of "" unless the header is in the same directory as the source file.
 * Tools like `clangd` often auto-insert includes when they can, but they usually get the grouping
   and brackets wrong.
 * Always check that includes are only necessary for the file in which they are included. 
   Try to avoid excessive including especially in header files. Double check this when you remove 
   code.

# libcudf Data Structures

Application data in libcudf is contained in Columns and Tables, but there are a variety of other
data structures you will use when developing libcudf code.

## Views and Ownership

Resource ownership is an essential concept in libcudf. In short, an "owning" object owns a 
resource (such as device memory). It acquires that resource during construction and releases the 
resource in destruction ([RAII](https://en.cppreference.com/w/cpp/language/raii)). A "non-owning"
object does not own resources. Any class in libcudf with the `*_view` suffix is non-owning. For more
detail see the [`libcudf++` presentation.](https://docs.google.com/presentation/d/1zKzAtc1AWFKfMhiUlV5yRZxSiPLwsObxMlWRWz_f5hA/edit?usp=sharing)

libcudf functions typically take views as input (`column_view`, `table_view`, or `scalar_view`)
and produce `unique_ptr`s to owning objects as output. For example, 

```c++
std::unique_ptr<table> sort(table_view const& input);
```

## `rmm::device_memory_resource`<a name="memory_resource"></a>

libcudf Allocates all device memory via RMM memory resources (MR). See the 
[RMM documentation](https://github.com/rapidsai/rmm/blob/main/README.md) for details.

### Current Device Memory Resource

RMM provides a "default" memory resource for each device that can be accessed and updated via the
`rmm::mr::get_current_device_resource()` and `rmm::mr::set_current_device_resource(...)` functions, 
respectively. All memory resource parameters should be defaulted to use the return value of 
`rmm::mr::get_current_device_resource()`. 

## `cudf::column`

`cudf::column` is a core owning data structure in libcudf. Most libcudf public APIs produce either 
a `cudf::column` or a `cudf::table` as output. A `column` contains `device_buffer`s which own the 
device memory for the elements of a column and an optional null indicator bitmask. 

Implicitly convertible to `column_view` and `mutable_column_view`. 

Movable and copyable. A copy performs a deep copy of the column's contents, whereas a move moves 
the contents from one column to another.

Example:
```c++
cudf::column col{...};

cudf::column copy{col}; // Copies the contents of `col`
cudf::column const moved_to{std::move(col)}; // Moves contents from `col`

column_view v = moved_to; // Implicit conversion to non-owning column_view
// mutable_column_view m = moved_to; // Cannot create mutable view to const column
```

A `column` may have nested (child) columns, depending on the data type of the column. For example,
`LIST`, `STRUCT`, and `STRING` type columns.

### `cudf::column_view`

`cudf::column_view` is a core non-owning data structure in libcudf. It is an immutable, 
non-owning view of device memory as a column. Most libcudf public APIs take views as inputs.

A `column_view` may be a view of a "slice" of a column. For example, it might view rows 75-150 of a 
column with 1000 rows. The `size()` of this `column_view` would be `75`, and accessing index `0` of 
the view would return the element at index `75` of the owning `column`. Internally, this is 
implemented by storing in the view a pointer, an offset, and a size. `column_view::data<T>()` 
returns a pointer iterator to `column_view::head<T>() + offset`.

### `cudf::mutable_column_view`

A *mutable*, non-owning view of device memory as a column. Used for detail APIs and (rare) public
APIs that modify columns in place.

### `cudf::column_device_view`

An immutable, non-owning view of device data as a column of elements that is trivially copyable and 
usable in CUDA device code. Used to pass `column_view` data as input to CUDA kernels and device 
functions (including Thrust algorithms)

### `cudf::mutable_column_device_view`

A mutable, non-owning view of device data as a column of elements that is trivially copyable and 
usable in CUDA device code. Used to pass `column_view` data to be modified on the device by CUDA
kernels and device functions (including Thrust algorithms).

## `cudf::table`

Owning class for a set of `cudf::column`s all with equal number of elements. This is the C++ 
equivalent to a data frame. 

Implicitly convertible to `cudf::table_view` and `cudf::mutable_table_view`

Movable and copyable. A copy performs a deep copy of all columns, whereas a move moves all columns 
from one table to another.

### `cudf::table_view`

An *immutable*, non-owning view of a table. 

### `cudf::mutable_table_view`

A *mutable*, non-owning view of a table. 

## `cudf::scalar`

A `cudf::scalar` is an object that can represent a singular, nullable value of any of the types 
currently supported by cudf. Each type of value is represented by a separate type of scalar class 
which are all derived from `cudf::scalar`. e.g. A `numeric_scalar` holds a single numerical value, 
a `string_scalar` holds a single string. The data for the stored value resides in device memory.

|Value type|Scalar class|Notes|
|-|-|-|
|fixed-width|`fixed_width_scalar<T>`| `T` can be any fixed-width type|
|numeric|`numeric_scalar<T>` | `T` can be `int8_t`, `int16_t`, `int32_t`, `int_64_t`, `float` or `double`|
|fixed-point|`fixed_point_scalar<T>` | `T` can be `numeric::decimal32` or `numeric::decimal64`|
|timestamp|`timestamp_scalar<T>` | `T` can be `timestamp_D`, `timestamp_s`, etc.|
|duration|`duration_scalar<T>` | `T` can be `duration_D`, `duration_s`, etc.|
|string|`string_scalar`| This class object is immutable|

### Construction
`scalar`s can be created using either their respective constructors or using factory functions like 
`make_numeric_scalar()`, `make_timestamp_scalar()` or `make_string_scalar()`. 

### Casting
All the factory methods return a `unique_ptr<scalar>` which needs to be statically downcasted to 
its respective scalar class type before accessing its value. Their validity (nullness) can be 
accessed without casting. Generally, the value needs to be accessed from a function that is aware 
of the value type e.g. a functor that is dispatched from `type_dispatcher`. To cast to the 
requisite scalar class type given the value type, use the mapping utility `scalar_type_t` provided 
in `type_dispatcher.hpp` : 

```c++
//unique_ptr<scalar> s = make_numeric_scalar(...);

using ScalarType = cudf::scalar_type_t<T>;
// ScalarType is now numeric_scalar<T>
auto s1 = static_cast<ScalarType *>(s.get());
```

### Passing to device
Each scalar type has a corresponding non-owning device view class which allows access to the value 
and its validity from the device. This can be obtained using the function 
`get_scalar_device_view(ScalarType s)`. Note that a device view is not provided for a base scalar 
object, only for the derived typed scalar class objects.

# libcudf++ API and Implementation

## Streams

CUDA streams are not yet exposed in external libcudf APIs. However, in order to ease the transition 
to future use of streams, all libcudf APIs that allocate device memory or execute a kernel should be 
implemented using asynchronous APIs on the default stream (e.g., stream 0).

The recommended pattern for doing this is to make the definition of the external API invoke an 
internal API in the `detail` namespace. The internal `detail` API has the same parameters as the 
public API, plus a `rmm::cuda_stream_view` parameter at the end defaulted to 
`rmm::cuda_stream_default`. The implementation should be wholly contained in the `detail` API 
definition and use only asynchronous versions of CUDA APIs with the stream parameter.

In order to make the `detail` API callable from other libcudf functions, it should be exposed in a 
header placed in the `cudf/cpp/include/detail/` directory.

For example:

```c++
// cpp/include/cudf/header.hpp
void external_function(...);

// cpp/include/cudf/detail/header.hpp
namespace detail{
void external_function(..., rmm::cuda_stream_view stream = rmm::cuda_stream_default)
} // namespace detail

// cudf/src/implementation.cpp
namespace detail{
    // defaulted stream parameter
    void external_function(..., rmm::cuda_stream_view stream){
        // implementation uses stream w/ async APIs
        rmm::device_buffer buff(...,stream);
        CUDA_TRY(cudaMemcpyAsync(...,stream.value()));
        kernel<<<..., stream>>>(...);
        thrust::algorithm(rmm::exec_policy(stream), ...);
    }
} // namespace detail

void external_function(...){
    detail::external_function(...);
}
```

**Note:** It is important to synchronize the stream if *and only if* it is necessary. For example,
when a non-pointer value is returned from the API that is the result of an asynchronous 
device-to-host copy, the stream used for the copy should be synchronized before returning. However,
when a column is returned, the stream should not be synchronized because doing so will break 
asynchrony if and when we add an asynchronous API to libcudf.

**Note:** `cudaDeviceSynchronize()` should *never* be used.
 This limits the ability to do any multi-stream/multi-threaded work with libcudf APIs.

 ### Stream Creation

There may be times in implementing libcudf features where it would be advantageous to use streams 
*internally*, i.e., to accomplish overlap in implementing an algorithm. However, dynamically 
creating a stream can be expensive. RMM has a stream pool class to help avoid dynamic stream 
creation. However, this is not yet exposed in libcudf, so for the time being, libcudf features 
should avoid creating streams (even if it is slightly less efficient). It is a good idea to leave a
`// TODO:` note indicating where using a stream would be beneficial.

## Memory Allocation

Device [memory resources](#memory_resource) are used in libcudf to abstract and control how device 
memory is allocated. 

### Output Memory

Any libcudf API that allocates memory that is *returned* to a user must accept a pointer to a 
`device_memory_resource` as the last parameter. Inside the API, this memory resource must be used
to allocate any memory for returned objects. It should therefore be passed into functions whose
outputs will be returned. Example:

```c++
// Returned `column` contains newly allocated memory, 
// therefore the API must accept a memory resource pointer
std::unique_ptr<column> returns_output_memory(
  ..., rmm::device_memory_resource * mr = rmm::mr::get_current_device_resource());

// This API does not allocate any new *output* memory, therefore
// a memory resource is unnecessary
void does_not_allocate_output_memory(...);                                              
```

### Temporary Memory

Not all memory allocated within a libcudf API is returned to the caller. Often algorithms must 
allocate temporary, scratch memory for intermediate results. Always use the default resource
obtained from `rmm::mr::get_current_device_resource()` for temporary memory allocations. Example:

```c++
rmm::device_buffer some_function(
  ..., rmm::mr::device_memory_resource mr * = rmm::mr::get_current_device_resource()) {
    rmm::device_buffer returned_buffer(..., mr); // Returned buffer uses the passed in MR
    ...
    rmm::device_buffer temporary_buffer(...); // Temporary buffer uses default MR
    ...
    return returned_buffer;
}
```

### Memory Management

libcudf code generally eschews raw pointers and direct memory allocation. Use RMM classes built to
use `device_memory_resource`(*)s for device memory allocation with automated lifetime management.

#### `rmm::device_buffer`
Allocates a specified number of bytes of untyped, uninitialized device memory using a 
`device_memory_resource`. If no resource is explicitly provided, uses 
`rmm::mr::get_current_device_resource()`. 

`rmm::device_buffer` is copyable and movable. A copy performs a deep copy of the `device_buffer`'s 
device memory, whereas a move moves ownership of the device memory from one `device_buffer` to 
another.

```c++
// Allocates at least 100 bytes of uninitialized device memory 
// using the specified resource and stream
rmm::device_buffer buff(100, stream, mr); 
void * raw_data = buff.data(); // Raw pointer to underlying device memory

rmm::device_buffer copy(buff); // Deep copies `buff` into `copy`
rmm::device_buffer moved_to(std::move(buff)); // Moves contents of `buff` into `moved_to`

custom_memory_resource *mr...;
rmm::device_buffer custom_buff(100, mr); // Allocates 100 bytes from the custom_memory_resource
```

#### `rmm::device_scalar<T>`
Allocates a single element of the specified type initialized to the specified value. Use this for 
scalar input/outputs into device kernels, e.g., reduction results, null count, etc. This is 
effectively a convenience wrapper around a `rmm::device_vector<T>` of length 1.

```c++
// Allocates device memory for a single int using the specified resource and stream
// and initializes the value to 42
rmm::device_scalar<int> int_scalar{42, stream, mr}; 

// scalar.data() returns pointer to value in device memory
kernel<<<...>>>(int_scalar.data(),...);

// scalar.value() synchronizes the scalar's stream and copies the 
// value from device to host and returns the value
int host_value = int_scalar.value();
```

#### `rmm::device_vector<T>`

Allocates a specified number of elements of the specified type. If no initialization value is 
provided, all elements are default initialized (this incurs a kernel launch).

**Note**: `rmm::device_vector<T>` is not yet updated to use `device_memory_resource`s, but support 
is forthcoming. Likewise, `device_vector` operations cannot be stream ordered.

#### `rmm::device_uvector<T>`

Similar to a `device_vector`, allocates a contiguous set of elements in device memory but with key 
differences:
- As an optimization, elements are uninitialized and no synchronization occurs at construction.
This limits the types `T` to trivially copyable types.
- All operations are stream ordered (i.e., they accept a `cuda_stream_view` specifying the stream 
on which the operation is performed).

```c++
cuda_stream s;
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the
// default resource
rmm::device_uvector<int32_t> v(100, s);
// Initializes the elements to 0
thrust::uninitialized_fill(thrust::cuda::par.on(s.value()), v.begin(), v.end(), int32_t{0}); 

rmm::mr::device_memory_resource * mr = new my_custom_resource{...};
// Allocates uninitialized storage for 100 `int32_t` elements on stream `s` using the resource `mr`
rmm::device_uvector<int32_t> v2{100, s, mr}; 
```

## Input/Output Style<a name="inout_style"></a>

The preferred style for how inputs are passed in and outputs are returned is the following:
-   Inputs
	- Columns:
		- `column_view const&`
	- Tables:
		- `table_view const&`
    - Scalar:
        - `scalar const&`
    - Everything else:
       - Trivial or inexpensively copied types
          - Pass by value
       - Non-trivial or expensive to copy types
          - Pass by `const&`
-   In/Outs  
	- Columns:
		- `mutable_column_view&`
	- Tables:
		- `mutable_table_view&`
    - Everything else:
        - Pass by via raw pointer
-   Outputs 
	- Outputs should be *returned*, i.e., no output parameters
	- Columns:
		- `std::unique_ptr<column>`
	- Tables:
		- `std::unique_ptr<table>`
    - Scalars:
        - `std::unique_ptr<scalar>`


### Multiple Return Values

Sometimes it is necessary for functions to have multiple outputs. There are a few ways this can be 
done in C++ (including creating a `struct` for the output). One convenient way to do this is 
using `std::tie`  and `std::make_pair`. Note that objects passed to `std::make_pair` will invoke 
either the copy constructor or the move constructor of the object, and it may be preferable to move 
non-trivially copyable objects (and required for types with deleted copy constructors, like 
`std::unique_ptr`).

```c++
std::pair<table, table> return_two_tables(void){
  cudf::table out0;
  cudf::table out1;
  ...
  // Do stuff with out0, out1
  
  // Return a std::pair of the two outputs
  return std::make_pair(std::move(out0), std::move(out1));
}

cudf::table out0;
cudf::table out1;
std::tie(out0, out1) = cudf::return_two_outputs();
```

Note:  `std::tuple`  _could_  be used if not for the fact that Cython does not support 
`std::tuple`. Therefore, libcudf APIs must use `std::pair`, and are therefore limited to return 
only two objects of different types. Multiple objects of the same type may be returned via a 
`std::vector<T>`.

## Iterator-based interfaces

Increasingly, libcudf is moving toward internal (`detail`) APIs with iterator parameters rather 
than explicit `column`/`table`/`scalar` parameters. As with STL, iterators enable generic 
algorithms to be applied to arbitrary containers. A good example of this is `cudf::copy_if_else`. 
This function takes two inputs, and a Boolean mask. It copies the corresponding element from the 
first or second input depending on whether the mask at that index is `true` or `false`. Implementing
`copy_if_else` for all combinations of `column` and `scalar` parameters is simplified by using
iterators in the `detail` API.

```c++
template <typename FilterFn, typename LeftIter, typename RightIter>
std::unique_ptr<column> copy_if_else(
  bool nullable,
  LeftIter lhs_begin,
  LeftIter lhs_end,
  RightIter rhs,
  FilterFn filter,
  ...);
```
`LeftIter` and `RightIter` need only implement the necessary interface for an iterator. libcudf 
provides a number of iterator types and utilities that are useful with iterator-based APIs from 
libcudf as well as Thrust algorithms. Most are defined in `include/detail/iterator.cuh`. 

### Pair iterator

The pair iterator is used to access elements of nullable columns as a pair containing an element's 
value and validity. `cudf::detail::make_pair_iterator` can be used to create a pair iterator from a 
`column_device_view` or a `cudf::scalar`. `make_pair_iterator` is not available for 
`mutable_column_device_view`.

### Null-replacement iterator

This iterator replaces the null/validity value for each element with a specified constant (`true` or
`false`). Created using `cudf::detail::make_null_replacement_iterator`.

### Validity iterator

This iterator returns the validity of the underlying element (`true` or `false`). Created using 
`cudf::detail::make_validity_iterator`.

### Index-normalizing iterators

The proliferation of data types supported by libcudf can result in long compile times. One area
where compile time was a problem is in types used to store indices, which can be any integer type.
The "Indexalator", or index-normalizing iterator (`include/cudf/detail/indexalator.cuh`), can be 
used for index types (integers) without requiring a type-specific instance. It can be used for any 
iterator interface for reading an array of integer values of type `int8`, `int16`, `int32`, 
`int64`, `uint8`, `uint16`, `uint32`, or `uint64`. Reading specific elements always return a 
`cudf::size_type` integer.

Use the `indexalator_factory` to create an appropriate input iterator from a column_view. Example 
input iterator usage:

```c++
auto begin = indexalator_factory::create_input_iterator(gather_map);
auto end   = begin + gather_map.size();
auto result = detail::gather( source, begin, end, IGNORE, stream, mr );
```

Example output iterator usage:

```c++
auto result_itr = indexalator_factory::create_output_iterator(indices->mutable_view());
thrust::lower_bound(rmm::exec_policy(stream),
                    input->begin<Element>(),
                    input->end<Element>(),
                    values->begin<Element>(),
                    values->end<Element>(),
                    result_itr,
                    thrust::less<Element>());
```

## Namespaces

### External
All public libcudf APIs should be placed in the `cudf` namespace. Example:
```c++
namespace cudf{
   void public_function(...);
} // namespace cudf
```

The top-level `cudf` namespace is sufficient for most of the public API. However, to logically 
group a broad set of functions, further namespaces may be used. For example, there are numerous 
functions that are specific to columns of Strings. These functions reside in the `cudf::strings::` 
namespace. Similarly, functionality used exclusively for unit testing is in the `cudf::test::` 
namespace. 

### Internal

Many functions are not meant for public use, so place them in either the `detail` or an *anonymous* 
namespace, depending on the situation.

#### `detail` namespace

Functions or objects that will be used across *multiple* translation units (i.e., source files), 
should be exposed in an internal header file and placed in the `detail` namespace. Example:

```c++
// some_utilities.hpp
namespace cudf{
namespace detail{
void reusable_helper_function(...);
} // namespace detail
} // namespace cudf
```

#### Anonymous namespace

Functions or objects that will only be used in a *single* translation unit should be defined in an 
*anonymous* namespace in the source file where it is used. Example:

```c++
// some_file.cpp
namespace{
void isolated_helper_function(...);
} // anonymous namespace
```

[**Anonymous namespaces should *never* be used in a header file.**](https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file) 

# Error Handling

libcudf follows conventions (and provides utilities) enforcing compile-time and run-time 
conditions and detecting and handling CUDA errors. Communication of errors is always via C++ 
exceptions.

## Runtime Conditions

Use the `CUDF_EXPECTS` macro to enforce runtime conditions necessary for correct execution.

Example usage:
```c++
CUDF_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");
```

The first argument is the conditional expression expected to resolve to  `true`  under normal 
conditions. If the conditional evaluates to  `false`, then an error has occurred and an instance of  `cudf::logic_error` is thrown. The second argument to  `CUDF_EXPECTS` is a short description of the 
error that has occurred and is used for the exception's `what()` message. 

There are times where a particular code path, if reached, should indicate an error no matter what. 
For example, often the `default` case of a `switch` statement represents an invalid alternative. 
Use the `CUDF_FAIL` macro for such errors. This is effectively the same as calling 
`CUDF_EXPECTS(false, reason)`.

Example:
```c++
CUDF_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

Use the `CUDA_TRY` macro to check for the successful completion of CUDA runtime API functions. This 
macro throws a `cudf::cuda_error` exception if the CUDA API return value is not `cudaSuccess`. The 
thrown exception includes a description of the CUDA error code in it's  `what()`  message.

Example:

```c++
CUDA_TRY( cudaMemcpy(&dst, &src, num_bytes) );
```

## Compile-Time Conditions

Use `static_assert` to enforce compile-time conditions. For example,

```c++
template <typename T>
void trivial_types_only(T t){
   static_assert(std::is_trivial<T>::value, "This function requires a trivial type.");
...
}
```

# Data Types

Columns may contain data of a number of types (see `enum class type_id` in `include/cudf/types.hpp`)

 * Numeric data: signed and unsigned integers (8-, 16-, 32-, or 64-bit), floats (32- or 64-bit), and
   Booleans (8-bit).
 * Timestamp data with resolution of days, seconds, milliseconds, microseconds, or nanoseconds.
 * Duration data with resolution of days, seconds, milliseconds, microseconds, or nanoseconds.
 * Decimal fixed-point data (32- or 64-bit).
 * Strings
 * Dictionaries
 * Lists of any type
 * Structs of columns of any type

Most algorithms must support columns of any data type. This leads to complexity in the code, and 
is one of the primary challenges a libcudf developer faces. Sometimes we develop new algorithms with
gradual support for more data types to make this easier. Typically we start with fixed-width data
types such as numeric types and timestamps/durations, adding support for nested types later.

Enabling an algorithm differently for different types uses either template specialization or SFINAE,
as discussed in [Specializing Type-Dispatched Code Paths](#specializing-type-dispatched-code-paths).

# Type Dispatcher

libcudf stores data (for columns and scalars) "type erased" in `void*` device memory. This 
*type-erasure* enables interoperability with other languages and type systems, such as Python and 
Java. In order to determine the type, libcudf algorithms must use the run-time information stored 
in the column `type()` to reconstruct the data type `T` by casting the `void*` to the appropriate 
`T*`.

This so-called *type dispatch* is pervasive throughout libcudf. The `type_dispatcher` is a 
central utility that automates the process of mapping the runtime type information in `data_type` 
to a concrete C++ type.

At a high level, you call the `type_dispatcher` with a `data_type` and a function object (also 
known as a *functor*) with an `operator()` template. Based on the value of `data_type::id()`, the 
type dispatcher invokes the corresponding instantiation of the `operator()` template. 

This simplified example shows how the value of `data_type::id()` determines which instantiation of 
the `F::operator()` template is invoked.

```c++
template <typename F>
void type_dispatcher(data_type t, F f){
    switch(t.id())
       case type_id::INT32: f.template operator()<int32_t>()
       case type_id::INT64: f.template operator()<int64_t>()
       case type_id::FLOAT: f.template operator()<float>()
       ...
}
```

The following example shows a function object called `size_of_functor` that returns the size of the 
dispatched type.

```c++
struct size_of_functor{
  template <typename T>
  int operator()(){ return sizeof(T); }
};

cudf::type_dispatcher(data_type{type_id::INT8}, size_of_functor{});  // returns 1
cudf::type_dispatcher(data_type{type_id::INT32}, size_of_functor{});  // returns 4
cudf::type_dispatcher(data_type{type_id::FLOAT64}, size_of_functor{});  // returns 8
```

By default, `type_dispatcher` uses `cudf::type_to_id<t>` to provide the mapping of `cudf::type_id` 
to dispatched C++ types. However, this mapping may be customized by explicitly specifying a 
user-defined trait for the `IdTypeMap`. For example, to always dispatch `int32_t` for all values of 
`cudf::type_id`:

```c++
template<cudf::type_id t> struct always_int{ using type = int32_t; }

// This will always invoke `operator()<int32_t>`
cudf::type_dispatcher<always_int>(data_type, f);
```

## Avoid Multiple Type Dispatch

Avoid multiple type-dispatch if possible. The compiler creates a code path for every type 
dispatched, so a second-level type dispatch results in quadratic growth in compilation time and 
object code size. As a large library with many types and functions, we are constantly working to
reduce compilation time and code size.

## Specializing Type-Dispatched Code Paths

It is often necessary to customize the dispatched `operator()` for different types. This can be 
done in several ways.

The first method is to use explicit, full template specialization. This is useful for specializing 
behavior for single types. The following example function object prints `"int32_t"` or `"double"` 
when invoked with either of those types, or `"unhandled type"` otherwise.

```c++
struct type_printer {
template <typename ColumnType>
void operator()() { std::cout << "unhandled type\n"; }
};

// Due to a bug in g++, explicit member function specializations need to be
// defined outside of the class definition
template <>
void type_printer::operator()<int32_t>() { std::cout << "int32_t\n"; }

template <>
void type_printer::operator()<double>() { std::cout << "double\n"; }
```

The second method is to use [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) with 
`std::enable_if_t`. This is useful to partially specialize for a set of types with a common trait. 
The following example functor prints `integral` or `floating point` for integral or floating point
types, respectively.

```c++
struct integral_or_floating_point {
template <typename ColumnType,
          std::enable_if_t<not std::is_integral<ColumnType>::value and
                           not std::is_floating_point<ColumnType>::value>* = nullptr> 
void operator()() { std::cout << "neither integral nor floating point\n"; }

template <typename ColumnType,
          std::enable_if_t<std::is_integral<ColumnType>::value>* = nullptr>
void operator()() { std::cout << "integral\n"; }

template < typename ColumnType,
           std::enable_if_t<std::is_floating_point<ColumnType>::value>* = nullptr> 
void operator()() { std::cout << "floating point\n"; }
};
```

For more info on SFINAE with `std::enable_if`, [see this post](https://eli.thegreenplace.net/2014/sfinae-and-enable_if).

There are a number of traits defined in `include/cudf/utilities/traits.hpp` that are useful for 
partial specialization of dispatched function objects. For example `is_numeric<T>()` can be used to 
specialize for any numeric type.

# Variable-Size and Nested Data Types

libcudf supports a number of variable-size and nested data types, including strings, lists, and 
structs. 
 
 * `string`: Simply a character string, but a column of strings may have a different-length string 
   in each row.
 * `list`: A list of elements of any type, so a column of lists of integers has rows with a list of 
   integers, possibly of a different length, in each row. 
 * `struct`: In a column of structs, each row is a structure comprising one or more fields. These
   fields are stored in structure-of-arrays format, so that the column of structs has a nested
   column for each field of the structure. 

As the heading implies, list and struct columns may be nested arbitrarily. One may create a column 
of lists of structs, where the fields of the struct may be of any type, including strings, lists and 
structs. Thinking about deeply nested data types can be confusing for column-based data, even with 
experience. Therefore it is important to carefully write algorithms, and to test and document them
well.

## List columns

In order to represent variable-width elements, libcudf columns contain a vector of child columns.
For list columns, the parent column's type is `LIST` and contains no data, but its size represents
the number of lists in the column, and its null mask represents the validity of each list element.
The parent has two children. 

1. A non-nullable column of `INT32` elements that indicates the offset to the beginning of each list
   in a dense column of elements.
2. A column containing the actual data and optional null mask for all elements of all the lists 
   packed together.
   
With this representation, `data[offsets[i]]` is the first element of list `i`, and the size of list
`i` is given by `offsets[i+1] - offsets[i]`.

Note that the data may be of any type, and therefore the data column may itself be a nested column
of any type. Note also that not only is each list nullable (using the null mask of the parent), but
each list element may be nullable. So you may have a lists column with null row 3, and also null
element 2 of row 4.

The underlying data for a lists column is always bundled into a single leaf column at the very 
bottom of the hierarchy (ignoring structs, which conceptually "reset" the root of the hierarchy), 
regardless of the level of nesting. So a `List<List<List<List<int>>>>>` column has a single `int` 
column at the very bottom. The following is a visual representation of this.

```
lists_column = { {{{1, 2}, {3, 4}}, NULL}, {{{10, 20}, {30, 40}}, {{50, 60, 70}, {0}}} }

   List<List<List<int>>>  (2 rows):
   Length : 2
   Offsets : 0, 2, 4
   Children :
      List<List<int>>:
      Length : 4
      Offsets : 0, 2, 2, 4, 6
      Null count: 1
        1101
      Children :
        List<int>:
        Length : 6
        Offsets : 0, 2, 4, 6, 8, 11, 12
        Children :
          Column of ints
          1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 0
```

This is related to [Arrow's "Variable-Size List" memory layout](https://arrow.apache.org/docs/format/Columnar.html?highlight=nested%20types#physical-memory-layout).

## Strings columns

Strings are represented in much the same way as lists, except that the data child column is always 
a non-nullable column of `INT8` data. The parent column's type is `STRING` and contains no data,
but its size represents the number of strings in the column, and its null mask represents the 
validity of each string. To summarize, the strings column children are:

1. A non-nullable column of `INT32` elements that indicates the offset to the beginning of each 
   string in a dense column of all characters.
2. A non-nullable column of `INT8` elements of all the characters across all the strings packed 
   together.

With this representation, `characters[offsets[i]]` is the first character of string `i`, and the 
size of string `i` is given by `offsets[i+1] - offsets[i]`. The following image shows an example of
this compound column representation of strings.

![strings](strings.png)

## Structs columns

Structs are represented similarly to lists, except that they have multiple child data columns.
The parent column's type is `STRUCT` and contains no data, but its size represents the number of 
structs in the column, and its null mask represents the validity of each struct element. The parent 
has `N + 1` children, where `N` is the number of fields in the struct. 

1. A non-nullable column of `INT32` elements that indicates the offset to the beginning of each 
   struct in each dense column of elements.
2. For each field, a column containing the actual field data and optional null mask for all elements
   of all the structs packed together.
   
With this representation, `child[0][offsets[i]]` is the first field of struct `i`, 
`child[1][offsets[i]]` is the second field of struct `i`, etc.

As defined in the [Apache Arrow specification](https://arrow.apache.org/docs/format/Columnar.html#struct-layout),
in addition to the struct column's null mask, each struct field column has its own optional null
mask. A struct field's validity can vary independently from the corresponding struct row. For
instance, a non-null struct row might have a null field. However, the fields of a null struct row
are deemed to be null as well. For example, consider a struct column of type 
`STRUCT<FLOAT32, INT32>`. If the contents are `[ {1.0, 2}, {4.0, 5}, null, {8.0, null} ]`, the
struct column's layout is as follows. (Note that null masks should be read from right to left.)

```
{
  type = STRUCT
  null_mask = [1, 1, 0, 1]
  null_count = 1
  children = {
    {   
      type = FLOAT32
      data =       [1.0, 4.0, X, 8.0]
      null_mask  = [  1,   1, 0,   1]
      null_count = 1
    },  
    {   
      type = INT32
      data =       [2, 5, X, X]
      null_mask  = [1, 1, 0, 0]
      null_count = 2
    }  
  }   
}
```

The last struct row (index 3) is not null, but has a null value in the INT32 field. Also, row 2 of 
the struct column is null, making its corresponding fields also null. Therefore, bit 2 is unset in 
the null masks of both struct fields.

## Dictionary columns

Dictionaries provide an efficient way to represent low-cardinality data by storing a single copy 
of each value. A dictionary comprises a column of sorted keys and a column containing an index into 
the keys column for each row of the parent column. The keys column may have any libcudf data type, 
such as a numerical type or strings. The indices represent the corresponding positions of each 
element's value in the keys. The indices child column can have any unsigned integer type 
(`UINT8`, `UINT16`, `UINT32`, or `UINT64`).

## Nested column challenges

The first challenge with nested columns is that it is effectively impossible to do any operation 
that modifies the length of any string or list in place. For example, consider trying to append the 
character `'a'` to the end of each string. This requires dynamically resizing the characters column
to allow inserting `'a'` at the end of each string, and then modifying the offsets column to 
indicate the new size of each element. As a result, every operation that can modify the strings or
lists in a column must be done out-of-place.

The second challenge is that in an out-of-place operation on a strings column, unlike with fixed-
width elements, the size of the output cannot be known *a priori*. For example, consider scattering 
into a column of strings:

```c++
destination:    {"this", "is", "a", "column", "of", "strings"}
scatter_map:    {1, 3, 5}
scatter_values: {"red", "green", "blue"}

result:         {"this", "red", "a", "green", "of", "blue"}
```

In this example, the strings "red", "green", and "blue" will respectively be scattered into
positions `1`, `3`, and `5` of `destination`. Recall from above that this operation cannot be done 
in place, therefore `result` will be generated by selectively copying strings from `destination` and
`scatter_values`. Notice that `result`'s child column of characters requires storage for `19`
characters. However, there is no way to know ahead of time that `result` will require `19`
characters. Therefore, most operations that produce a new output column of strings use a two-phase
approach:

1. Determine the number and size of each string in the result. This amounts to materializing the
   output offsets column.
2. Allocate sufficient storage for all of the output characters and materialize each output string.

In scatter, the first phase consists of using the `scatter_map` to determine whether string `i` in
the output will come from `destination` or from `scatter_values` and use the corresponding size(s) 
to materialize the offsets column and determine the size of the output. Then, in the second phase, 
sufficient storage is allocated for the output's characters, and then the characters are filled 
with the corresponding strings from either `destination` or `scatter_values`.

## Nested Type Views

libcudf provides view types for nested column types as well as for the data elements within them.

### `cudf::strings_column_view` and `cudf::string_view`

`cudf::strings_column_view` is a view of a strings column, like `cudf::column_view` is a view of 
any `cudf::column`. `cudf::string_view` is a view of a single string, and therefore 
`cudf::string_view` is the data type of a `cudf::column` of type `STRING` just like `int32_t` is the
data type for a `cudf::column` of type `INT32`. As it's name implies, this is a read-only object 
instance that points to device memory inside the strings column. It's lifespan is the same (or less)
as the column it views.

Use the `column_device_view::element` method to access an individual row element. Like any other
column, do not call `element()` on a row that is null. 

```c++
   cudf::column_device_view d_strings;
   ...
   if( d_strings.is_valid(row_index) ) {
      string_view d_str = d_strings.element<string_view>(row_index);
      ...
   }
```

A null string is not the same as an empty string. Use the `string_scalar` class if you need an 
instance of a class object to represent a null string.

The `string_view` contains comparison operators `<,>,==,<=,>=` that can be used in many cudf 
functions like `sort` without string-specific code. The data for a `string_view` instance is 
required to be [UTF-8](#UTF-8) and all operators and methods expect this encoding. Unless documented
otherwise, position and length parameters are specified in characters and not bytes. The class also
includes a `string_view::const_iterator` which can be used to navigate through individual characters
within the string.

`cudf::type_dispatcher` dispatches to the `string_view` data type when invoked on a `STRING` column.

#### UTF-8

The libcudf strings column only supports UTF-8 encoding for strings data. 
[UTF-8](https://en.wikipedia.org/wiki/UTF-8) is a variable-length character encoding wherein each 
character can be 1-4 bytes. This means the length of a string is not the same as its size in bytes.
For this reason, it is recommended to use the `string_view` class to access these characters for
most operations.

The `string_view.cuh` header also includes some utility methods for reading and writing 
(`to_char_utf8/from_char_utf8`) individual UTF-8 characters to/from byte arrays.

### `cudf::lists_column_view` and `cudf::lists_view`

`cudf::lists_column_view` is a view of a lists column. `cudf::list_view` is a view of a single list,
and therefore `cudf::list_view` is the data type of a `cudf::column` of type `LIST`.

`cudf::type_dispatcher` dispatches to the `list_view` data type when invoked on a `LIST` column.

### `cudf::structs_column_view` and `cudf::struct_view`

`cudf::structs_column_view` is a view of a structs column. `cudf::struct_view` is a view of a single
struct, and therefore `cudf::struct_view` is the data type of a `cudf::column` of type `STRUCT`.

`cudf::type_dispatcher` dispatches to the `struct_view` data type when invoked on a `STRUCT` column.

# cuIO: file reading and writing

cuIO is a component of libcudf that provides GPU-accelerated reading and writing of data file 
formats commonly used in data analytics, including CSV, Parquet, ORC, Avro, and JSON_Lines.

// TODO: add more detail and move to a separate file.
