# libcudf++ C++ Transition Guide
This document serves as a guide for the set of new features introduced revolving around `cudf::column` (colloquially known as `libucdf++`) as well as how to transition from from "old" `gdf_column` based APIs to the new types. 

This guide assumes familiarity with the concepts and types introduced in the [`libcudf++` presentation.](https://docs.google.com/presentation/d/1zKzAtc1AWFKfMhiUlV5yRZxSiPLwsObxMlWRWz_f5hA/edit?usp=sharing) 

# Directory Structure and File Naming

External APIs should be grouped based on functionality into an appropriately titled header file  in `cudf/cpp/include/cudf/`. For example,  `cudf/cpp/include/cudf/copying.hpp` contains the APIs for functions related to copying from one column to another. Note the  `.hpp`  file extension used to indicate a C++ header file. These header files should use the `#pragma once` include guard. 

The naming of these headers should be consistent with the name of the folder that contains the source files that implement  the API. For example, the implementation of the APIs found in  `cudf/cpp/include/cudf/copying.hpp`  are located in  `cudf/src/copying`. Likewise, the unit tests for the APIs reside in `cudf/tests/copying/`. 

 ## Legacy Directories

When an old API is replaced by a new `libcudf++` API, it should be moved into a `legacy/` directory within it's current directory.

For example, when porting the implementation of `gather` in `cudf/src/copying/gather.cu`, the existing implementation should be moved to `cudf/src/copying/legacy/gather.cu`. Likewise, the header `cudf/cpp/include/cudf/copying.hpp` should be moved to `cudf/cpp/include/cudf/legacy/copying.hpp`. 


# libcudf++ Data Structures

## Views and Ownership

Resource ownership is an essential concept in `libcudf++`. In short, an "owning" object owns some resource (such as device memory). It acquires that resource during construction and releases the resource in destruction (RAII). A "non-owning" object does not own resources. Any class in `libcudf++` with the `*_view` suffix is non-owning. For more detail see the [`libcudf++` presentation.](https://docs.google.com/presentation/d/1zKzAtc1AWFKfMhiUlV5yRZxSiPLwsObxMlWRWz_f5hA/edit?usp=sharing) 


## `rmm::device_memory_resource`

Abstract interface class based on [`std::pmr::memory_resource`](https://en.cppreference.com/w/cpp/memory/memory_resource) for allocating device memory. 

Provides `allocate` and `deallocate` member functions for device memory allocation. Creating a new class that derives from `rmm::device_memory_resource` and overriding the `do_allocate` and `do_deallocate` pure-virtual functions allows customization of device memory allocation. 

### Default Memory Resources

RMM provides a "default" memory resource that can be accessed and updated via the `rmm::mr::get_default_resource()` and `rmm::mr::set_default_resource(...)` functions, respectively. All memory resource arguments should be defaulted to use the return value of `rmm::mr::get_default_resource()`. 


## `rmm::device_buffer`

The fundamental device memory owning class in `libcudf++`. 

Allocates non-typed, uninitialized device memory using a `device_memory_resource`. If no resource is explicitly provided, uses `rmm::mr::get_default_resource()`. 

Movable and copyable. A copy performs a deep copy of the `device_buffer`'s device memory, whereas a move moves ownership of the device memory from one `device_buffer` to another.

Example:
```c++
rmm::device_buffer buff(100); // Allocates at least 100 bytes of uninitialized device memory
void * raw_data = buff.data(); // Raw pointer to underlying device memory

rmm::device_buffer copy(buff); // Deep copies `buff` into `copy`
rmm::device_buffer moved_to(std::move(buff)); // Moves contents of `buff` into `moved_to`

custom_memory_resource mr;
rmm::device_buffer custom_buff(100, &mr); // Allocates 100 bytes from the custom memory resource
```

## `cudf::column`

Contains `device_buffer`s which own the device memory for the elements of a column and it's optional null indicator bitmask. 

Implicitly convertible to `column_view` and `mutable_column_view`. 

Movable and copyable. A copy performs a deep copy of the column's contents, whereas a move moves the contents from one column to another.

Example:
```c++
cudf::column col{...};

cudf::column copy{col}; // Copies the contents of `col`
cudf::column const moved_to{std::move(col)}; // Moves contents from `col`

column_view v = moved_to; // Implicit conversion to non-owning column_view
// mutable_column_view m = moved_to; // Cannot create mutable view to const column
```

### `cudf::column_view`

An immutable, non-owning view of device memory as a column. 

Trivially-copyable and should be passed by value. 

### `cudf::mutable_column_view`

A *mutable*, non-owning view of device memory as a column. 

Trivially-copyable and should be passed by value. 

## `cudf::table`

Owning class for a set of `cudf::column`s all with equal number of elements. This is the C++ corollary to a DataFrame. 

Implicitly convertible to `cudf::table_view` and `cudf::mutable_table_view`

Movable and copyable. A copy performs a deep copy of all columns, whereas a move moves all columns from one table to another.

### `cudf::table_view`

An *immutable*, non-owning view of a table. 

Trivially-copyable and should be passed by value. 

### `cudf::mutable_table_view`

A *mutable*, non-owning view of a table. 

Trivially-copyable and should be passed by value. 

# libcudf++ API
### Old libcudf API
```c++
/**
* @brief This is an example of an old API.
*
* @param input Immutable input column
* @param in_out Mutable column modified in place
* @param input_table Immutable input table
* @param in_out_table Mutable table modified in place
* @return gdf_column Newly allocated output column
**/

gdf_column  some_function(gdf_column const& input, 
                          gdf_column* in_out,
                          cudf::table const& input_table,
                          cudf::table* in_out_table);
```

### New libcudf API
```c++
/**
 * @brief This is an example of a new API.
 *
 * @param input Immutable view of input column
 * @param in_out Mutable view of column modified in place
 * @param input_table Immutable view of input table
 * @param in_out_table Mutable view of table modified in place
 * @param mr Memory resource used to allocate device memory for the returned
 * output column
 * @return std::unique_ptr<column> Newly allocated output column
 **/
std::unique_ptr<column> some_function(cudf::column_view input, 
                                      cudf::mutable_column_view in_out, 
                                      cudf::table_view input_table,
                                      cudf::mutable_table_view in_out_table,
                                      device_memory_resource* mr = rmm::get_default_resource());
```
## Input/Output Style<a name="inout_style"></a>

All `*_view` objects are trivially copyable and are intended to be passed by value.

The preferred style for how inputs are passed in and outputs are returned is the following:
-   Inputs
	- Columns:
		- `column_view`
	- Tables:
		- `table_view`
    - Everything else:
       - Trivial or inexpensively copied types
          - Pass by value
       - Non-trivial or expensive to copy types
          - Pass by `const&`
-   In/Outs  
	- Columns:
		- `mutable_column_view`
	- Tables:
		- `mutable_table_view`
    - Everything else:
        - Pass by via raw pointer
-   Outputs 
	- Outputs should be *returned*, i.e., no output parameters
	- Columns:
		- `std::unique_ptr<column>`
	- Tables:
		- `std::unique_ptr<table>`


### Multiple Return Values

Sometimes it is necessary for functions to have multiple outputs. There are a few ways this can be done in C++ (including creating a  `struct`  for the output). One convenient way to do this is using  `std::tie`  and  `std::make_pair`.

```c++
std::pair<table, table> return_two_tables(void){
  cudf::table out0;
  cudf::table out1;
  ...
  // Do stuff with out0, out1
  
  // Return a std::pair of the two outputs
  return std::make_pair(out0, out1);
}

cudf::table out0;
cudf::table out1;
std::tie(out0, out1) = cudf::return_two_outputs();
```

Note:  `std::tuple`  _could_  be used if not for the fact that Cython does not support  `std::tuple`. Therefore, libcudf APIs must use  `std::pair`, and are therefore limited to return only two objects of different types. Multiple objects of the same type may be returned via a  `std::vector<T>`.

## Namespaces


### External
All public libcudf APIs should be placed in the `cudf` namespace*. Example:
```c++
namespace cudf{
   void public_function(...);
} // namespace cudf
```

For most functions, the top-level `cudf` namespace is sufficient. However, for logically grouping a broad set of functions, further namespaces may be used. For example, there are numerous functions that are specific to columns of Strings. These functions are put in the `cudf::strings::` namespace. Similarly, functionality used exclusively for unit testing is placed in the `cudf::test::` namespace. 

### `experimental`
During the transition period, symbols in `libcudf++` that conflict with old symbol names should be placed in the `cudf::experimental` namespace to prevent collision with the old symbols, e.g., `cudf::experimental::table` and `cudf::experimental::type_dispatcher`. Once the transition is complete, the `experimental` namespace will be removed.

### Internal

Many functions are not meant for public use. Such functions should be placed in either the `detail` or an *anonymous* namespace depending on the situation.

#### `detail`

For functions or objects that will be used across *multiple* translation units (i.e., source files), they should be exposed in an internal header file and placed in the `detail` namespace. Example:
```c++
// some_utilities.hpp
namespace cudf{
namespace detail{
void reusable_helper_function(...);
} // namespace detail
} // namespace cudf
```
#### Anonymous 

If a function or class will only be used in a *single* translation unit, it should be put in an *anonymous* namespace within source file where it is used. Example:
```c++
// some_file.cpp
namespace{
void isolated_helper_function(...);
} // anonymous namespace
```

[**Anonymous namespaces should *never* be used in a header file.**](https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file) 

# Error Handling

## Runtime Conditions

For ensuring runtime conditions necessary for correct execution, the `CUDF_EXPECTS` macro should be used.

Example usage:
```c++
CUDF_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");
```

The first argument is the conditional expression expected to resolve to  `true`  under normal conditions. If the conditional evaluates to  `false`, then an error has occurred and an instance of  `cudf::logic_error`  is thrown. The second argument to  `CUDF_EXPECTS`  is a short description of the error that has occurred and is used for the exceptions `what()` message. 

There are times where a particular code path, if reached, should indicate an error no matter what. For example, in the  `default`  case of a  `switch`  statement where the only valid code paths are in one of the  `case`  statements.

For these cases, the  `CUDF_FAIL`  convenience macro should be used. This is effectively the same as doing  `CUDF_EXPECTS(false, reason)`.

Example:
```c++
CUDF_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

Checking for the succesful completion of CUDA runtime API functions should be done via the  `CUDA_TRY`  macro. This macro throws a `cudf::cuda_error` exception if the return value of the CUDA API does not return  `cudaSuccess`. The thrown exception will include a description of the CUDA error code that occurred in it's  `what()`  message.

Example:

```c++
CUDA_TRY( cudaMemcpy(&dst, &src, num_bytes) );
```


## Compile-Time Conditions

Some conditions can be verified at compile time. These should be done using `static_assert`. For example,

```c++
template <typename T>
void trivial_types_only(T t){
   static_assert(std::is_trivial<T>::value, "This function requires a trivial type.");
...
}
```


# Type Dispatcher

The device memory for a column's elements is stored in a `void*`. 
This is known as *type-erasure*, because the data's type we are pointing to is not known at compile time.
In order to determine the type, we must use the runtime information stored in the columns `type()`. 
We can then use that type information to reconstruct the data's type `T`, i.e., casting the `void*` to the appropriate `T*`.

This type "reconstruction" or *type dispatching* is pervasive throughout `libcudf`. 
The `type_dispatcher` is a centralized utility that automates the process of mapping the runtime type information in `data_type` to a concrete C++ type.

At a high level, you give the `type_dispatcher` a `data_type` and a function object (sometimes known as a *functor*) with an `operator()` template. 
Then, based on the value of `data_type::id()`, it will invoke the corresponding instantiation of the `operator()` template. 

This simplified example of how the `type_dispatcher` works shows how the value of `data_type::id()` determines which instantiation of the `F::operator()` template is invoked. 
```c++
template <typename F>
void type_dispatcher(data_type t, F f){
    switch(t.id())
       case INT32: f.template operator()<int32_t>()
       case INT64: f.template operator()<int64_t>()
       case FLOAT: f.template operator()<float>()
       ...
}
```

The below example shows a function object called `size_of_functor` that returns the size of the dispatched type:

```c++
struct size_of_functor{
template <typename T>
int operator()(){ return sizeof(T); }
};
cudf::type_dispatcher(data_type{INT8}, size_of_functor{});  // returns 1
cudf::type_dispatcher(data_type{INT32}, size_of_functor{});  // returns 4
cudf::type_dispatcher(data_type{FLOAT64}, size_of_functor{});  // returns 8
```

By default, the `type_dispatcher` uses `cudf::type_to_id<t>` to provide the mapping
of `cudf::type_id`s to dispatched C++ types.
However, this mapping may be customized by explicitly specifying a user-defined trait for the `IdTypeMap`.
For example, to always dispatch `int32_t` for all values of `cudf::type_id`:

```c++
template<cudf::type_id t> struct always_int{ using type = int32_t; }

// This will always invoke `operator()<int32_t>`
cudf::type_dispatcher<always_int>(data_type, f);
```

It is often necessary to customize the dispatched `operator()` for different types. 
This can be done in several ways.

The first method is to use explicit, full template specialization. This is useful
for specializing behavior for single types. For example, a function object that
prints `"int32_t"` or `"double"` when invoked with either of those types, else it
prints `"unhandled type"`:

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

A second method is to use SFINAE with `std::enable_if_t`. This is useful for
partially specializing for a set of types that share some property. For example, a
functor that prints `integral` or `floating point` for integral or floating
point types:

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
For more info on SFINAE and `std::enable_if`, see https://eli.thegreenplace.net/2014/sfinae-and-enable_if

There are a number of traits defined in `cpp/include/cudf/utilities/traits.hpp` that are useful for partially specializing dispatched function objects. 
For example `is_numeric<T>()` can be used to specialize for any numeric type.


# Testing

TBD



# Porting Guide

This section provides the high-level steps necessary to port an existing `libcudf` API into a `libcudf++` function. 

## Pull Request Process

Porting an old API should be broken up into two PRs:
1. Moving the old API/implementation to `legacy/`
    - This is [steps 1-3 below](#legacy_changes)
    - This PR should not introduce any functional changes, and therefore be trivial to review and merged quickly.
2. Creating and implementing the new API
    - This is [steps 4-6 below](#new_changes)
    - This PR should be opened as soon as steps 4 & 5 are complete (i.e., once the API is created, but before beginning on the implementation). This is to allow early feedback on the API and make sure it is solid before effort is put into implementation.

## C++ Changes

For example, given the example function `old_function`:
```c++
// cpp/include/cudf/utilities/old.hpp

namespace cudf{
/**
* @brief This is an example of an old API.
*
* @param input Immutable input column
* @param in_out Mutable column modified in place
* @param input_table Immutable input table
* @param in_out_table Mutable table modified in place
* @return gdf_column Newly allocated output column
**/

gdf_column  old_function(gdf_column const& input, 
                         gdf_column* in_out,
                         cudf::table const& input_table,
                         cudf::table* in_out_table);
} // namespace cudf
```

1. Move old header and source/tests into `legacy/` directories<a name="legacy_changes"></a>
    - `mkdir -p cudf/cpp/include/cudf/utilities/legacy`
    - `mkdir -p cudf/cpp/src/utilities/legacy`
    - `mkdir -p cudf/cpp/tests/utilities/legacy`
    - `git mv cudf/cpp/include/cudf/utilities/old.hpp cudf/cpp/include/cudf/utilities/legacy/`
    - `git mv cudf/cpp/src/utilities/old.cpp cudf/cpp/include/cudf/utilities/legacy/`
    - `git mv cudf/cpp/tests/utilities/old_tests.cpp cudf/cpp/tests/utilities/legacy/`
2. Update paths to `old.hpp`, `old.cpp`, and `old_tests.cpp` 
    - `cudf/cpp/CMakeLists.txt` 
    - `cudf/cpp/tests/CMakeLists.txt`
    - `cudf/conda/recipes/libcudf/meta.yaml`
    - Include paths
    - Cython include paths (see [Cython changes](#cython_changes))
3. Update test names
    - Rename `OLD_TESTS` to `LEGACY_OLD_TESTS` in `cudf/cpp/tests/CMakeLists.txt`
4. Create new header and source files<a name="new_changes"></a>
    - `touch cudf/cpp/include/cudf/utilities/new.hpp`
        - Remember to use `#pragma once` for the include guard
    - `touch cudf/cpp/src/utilities/new.cpp`
    - `touch cudf/cpp/tests/utilities/new_tests.cpp`
5. Update API to use new data structures
    - See [replacement guide](#replacements) for common replacements. 
    - Many old APIs still use output parameters (e.g., `gdf_column *`). These must be updated to *return* outputs as specified in [the section on input/output style](#inout_style).
    - Likewise, many old APIs are inconsistent with `const` correctness and how input parameters are passed, e.g., `gdf_column*` may be used for what should be an immutable input column. Use your best judgement to determine how the parameter is being used and select the appropriate `libucdf++` type accordingly.
6. Update implementation to use new data structures
    - See [replacement guide](#replacements) for replacements of commonly used functions.
    - Use this as an opportunity to improve and cleanup the implementation
    - Likely the most dramatic algorithmic change will be adding native string column support. See the [section on string support](#string_support).


Example of the completed port of `old_function` to `new_function`. 
```c++
// cpp/include/cudf/utilities/new.hpp
	
namespace cudf{
namespace experimental{
/**
 * @brief This is an example of a new API.
 *
 * @param input Immutable view of input column
 * @param in_out Mutable view of column modified in place
 * @param input_table Immutable view of input table
 * @param in_out_table Mutable view of table modified in place
 * @param mr Memory resource used to allocate device memory for the returned
 * output column
 * @return std::unique_ptr<column> Newly allocated output column
 **/
std::unique_ptr<column> new_function(cudf::column_view input, 
                                     cudf::mutable_column_view in_out, 
                                     cudf::table_view input_table,
                                     cudf::mutable_table_view in_out_table,
                                     device_memory_resource* mr = rmm::get_default_resource());
 } // namespace experimental
 } // namespace cudf
```

## Cython changes<a name="cython_changes"></a>

In the short-term, Python will continue to use legacy `libcudf` functions and data structures.
All that needs to be changed in Cython are the paths to legacy headers.
For example, after moving `cudf/copying.hpp` to `cudf/legacy/copying.hpp`, the following line in
`python/cudf/cudf/_lib/includes.copying.pxd` will be changed from:

```python
cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
```

to:

```python
cdef extern from "cudf/legacy/copying.hpp" namespace "cudf" nogil:
```

## Common Replacements<a name="replacements"></a>

### Data Structures
|          Old         |                New                |                      Notes                      |
|:--------------------:|:---------------------------------:|:-----------------------------------------------:|
|  `gdf_column const&` |        `cudf::column_view`        |                                                 |
|    `gdf_column *`    |    `cudf::mutable_column_view`    | Only use when in/out is actually needed (rare). |
|  `gdf_column const*` |        `cudf::column_view`        |                                                 |
| `cudf::table const&` |         `cudf::table_view`        |                                                 |
|    `cudf::table *`   |     `cudf::mutable_table_view`    |                                                 |
| `cudf::device_table` |     `cudf::table_device_view`     |                    Immutable                    |
| `cudf::device_table` | `cudf::mutable_table_device_view` |                     Mutable                     |
| `gdf_scalar`         | `cudf::scalar`                    |                     Not yet implemented         |

### Functions

|              Old             |                       New                       |                                     Notes                                    |
|:----------------------------:|:-----------------------------------------------:|:----------------------------------------------------------------------------:|
|       `gdf_is_valid()`       |        `*column_device_view::is_valid()`        |         Verifies existence of bitmask before  attempting to read it.         |
|    `bit_mask::is_valid()`    |    `*column_device_view::is_valid_nocheck()`    | Does not verify the existence of the bitmask  before attempting to read it.  |
| `gdf_vald_allocation_size()` |        `bitmask_allocation_size_bytes()`        |                                                                              |
|        `is_nullable()`       |               `*view::nullable()`               |                                                                              |
|         `has_nulls()`        |               `*view::has_nulls()`              |                                                                              |
|        `cudf::copy()`        | `column::column(const&)` `table::table(const&)` |                               Copy constructors                              |
|    `cudf::allocate_like()`   |      `cudf::experimental::allocate_like()`      |                      This applies to both columns/tables                     |
|     `cudf::empty_like()`     |        `cudf::experimental::empty_like()`       |                      This applies to both columns/tables                     |

## Strings Support<a name="string_support"></a>

One of the more significant changes in `libcudf++` is the addition of native support for columns of strings. 
Previously, string support in libcudf was accomplished via the `NVCategory` class.
See the [NVCategory](#nvcategory_changes) section for more detail. 
This class will no longer exist in `libcudf++`, but a replacement "dictionary" column type is planned for the future to replace the functionality of NVCategory.

All libcudf functions will now need to be able to operate on a column of string elements directly. 
This poses a set of unique challenges, as unlike all previous libcudf types, the elements in a string column are variable width. 
In order to represent variable width strings, libcudf uses a *compound* column (i.e., column with children).
The parent column's type is `STRING` and contains no data, but contains the count of the number of strings and the bitmask representing the validity of each string element.
The parent has two children.
The first is a non-nullable column of `INT32` elements that indicate the offset to beginning of each string in a dense column of characters.
The second is a non-nullable column of `INT8` elements of all the characters across all the strings packed together. With this representation, `characters[offsets[i]]` is the first character of string `i`, and the size of string `i` is given by `offsets[i+1] - offsets[i]`.

The image below shows an example of this compound column representation of strings:


![strings](strings.png)

The first challenge with strings columns is that it is effectively impossible to do any operation that modifies the length of any string in-place.
For example, consider trying to append the character `'a'` to the end of each string.
This would require dynamically resizing the characters column to allow inserting `'a'` at the end of each string, and then modifying the offsets column to indicate the new size of each element. As a result, every operation that can modify the strings in the column must be out-of-place. 

The second challenge is that in an out-of-place operation on a string column, unlike with fixed-width elements, the size of the output cannot be known *a priori*. 
For example, consider scattering into a column of strings:
```c++
destination:    {"this", "is", "a", "column", "of", "strings"}
scatter_map:    {1, 3, 5}
scatter_values: {"red", "green", "blue"}

result:         {"this", "red", "a", "green", "of", "blue"}
```

In this example, the strings `"red", "green", "blue"` will respectively be scattered into positions `1,3,5` of `destination`. 
Recall from above that this operation cannot be done in-place, therefore `result` will be generated by selectively copying strings from `destination` and `scatter_values`. 
Notice that `result`'s child column of characters will require storage for `19` characters. 
However, there is no way to know ahead of time that `result` will require `19` characters. 
Therefore, most operations that produce a new output column of strings use a two-phase approach:

1. First, determine the number and size of each string in the result
    - This amounts to materializing the output's offsets column 
2. Second, allocate sufficient storage for all of the output characters and materialize each output string

In scatter, the first phase consists of using the `scatter_map` to determine if string `i` in the output will come from `destination` or from `scatter_values` and use the corresponding size(s) to materialize the offsets column and determine the size of the output. Then, in the second phase, sufficient storage is allocated for the output's characters, and then the characters are filled with the corresponding strings from either `destination` or `scatter_values`.






### NVCategory<a name="nvcategory_changes"></a>

// TODO

