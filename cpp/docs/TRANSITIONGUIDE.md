## Overview

As libcudf transitions from a C API to C++ API, this is a list of guidance on what should be done for all new PRs against libcudf. 

## File Structure

In lieu of the monolithic `functions.h`, external function APIs should be grouped based on functionality into an appropriately 
titled header file `cudf/cpp/include/`. 
For example, `cudf/cpp/include/copying.hpp` contains the APIs for functions related to copying from one column to another. Note the `.hpp` file extension used to indicate a C++ header file. 

As existing functions are ported from a C to C++ API, new header files will need to be added to `cudf/cpp/include/`. For naming of these headers, it should be consistent with the name of the folder that contains the implementation of the API. For example, the implementation of the APIs found in `cudf/cpp/include/copying.hpp` are located in `cudf/src/copying`. 


# API Design

## Namespaces

All external APIs should be placed in the `cudf` namespace. For example:

```
namespace cudf{

  void some_function(...);

} // namespace cudf

```

### `detail` Namespace

Utility functions that are for use only in libcudf internals, but can be reused in many places should be placed in the `cudf::detail` namespace. For example:

```
namespace cudf{
namespace detail{
void reusable_utility_function(...);
} // namespace detail
} // namespace cudf
```

### Anonymous Namespaces

All functions specific to a translation unit should be placed in an anonymous namespace.

Example:
```
namespace {
void some_specific_helper_function(...);
} // namespace
```

## Data Structures

### `cudf::column`

A replacement abstraction for `gdf_column` is actively being designed: https://github.com/rapidsai/cudf/issues/1443

Until that design is complete, continue to use `gdf_column`s as usual. 

### `cudf::table`

It is common for an API to work on a set of `N` `gdf_column`s of equal size which required an API to do something like the following: 

```
void some_multi_col_function(gdf_column * cols[], int num_cols);
```

A simple wrapper class called `cudf::table` is defined in `cudf/cpp/src/table/table.hpp` to provide a convenient abstraction for the above use case. This class should be used any time a function is operating on a collection of `gdf_column`s of equal size.

For example, the above becomes:
```
void some_multi_col_function(cudf::table & table);
```

## Input/Output Style

The preferred style for how inputs are passed in and outputs are returned is the following:

* Inputs 
   * Expensive to copy types --- pass by constant reference
   * Inexpensive to copy types --- pass by value
* In/Outs - Pass via pointer
* Outputs - Returned by value
   * Returning outputs by value will not incur a copy in most cases. For more details, see: https://stackoverflow.com/questions/12953127/what-are-copy-elision-and-return-value-optimization
   * **NOTE:** This may mean returning a `gdf_column` whose device memory was allocated within a `libcudf` function and it is the caller's responsibility to free the underlying memory. This is a temporary headache until a `cudf::column` abstraction is designed to allieviate it. 

For example:

```
namespace cudf{
column some_function( cudf::table const& input_table, cudf::table * input_output_table, int size){
  column col;
  // Do stuff to create output col
  ...
  // Return the result column of this function
  return col;
}
}
```

And the usage would be:
```
cudf::table input;
cudf::table in_out;
int size;
column output = cudf::some_function(input, &in_out, size);
```

### Multiple Return Values

Sometimes it is necessary for functions to have multiple outputs. There are a few ways this can be done in C++ (including creating a `struct` for the output). One convenient way to do this is using `std::tie` and `std::make_tuple`.

```
auto return_multiple_outputs(void){
  cudf::table out0;
  cudf::table out1;
  ...
  // Do stuff with out0, out1
  
  // Return a tuple of all the outputs
  return std::make_tuple(out0, out1);
}


cudf::table out0;
cudf::table out1;
std::tie(out0, out1) = cudf::return_multiple_outputs();
```


## Error Checking

The `gdf_error` codes are deprecated. Instead, `libcudf` functions should prefer throwing exceptions.

In place of the `GDF_REQUIRES` convenience macro, the `CUDF_EXPECTS` macro is provided in `cudf/cpp/src/utilities/error_utils.hpp`. 

Similar to `GDF_REQUIRES`, `CUDF_EXPECTS` will throw an exception if a condition is violated. The first argument is the conditional expression that can be evaluated as a `boolean` expression and is expected to resolve to `true` under normal conditions. If the conditional evaluates to `false`, then an error has occurred and an instance of `cudf::logic_error` is thrown. The second argument to `CUDF_EXPECTS` is a short description of the error that has occurred if the conditional is false. 

Example usage: 

```
CUDF_EXPECTS(lhs->dtype == rhs->dtype, "Column type mismatch");
```

There are times where a particular code path, if reached, should indicate an error no matter what. For example, in the `default` case of a `switch` statement where the only valid code paths are in one of the `case` statements. 

For these cases, the `CUDF_FAIL` convenience macro should be used. This is effectively the same as doing `CUDF_EXPECTS(false, reason)`. 

Example:

```
CUDF_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

Checking for the succesful completion of CUDA runtime API functions should continue to be done via the `CUDA_TRY` macro. Note that this macro has been updated to throw an exception of type `cudf::cuda_error` if the return value of the CUDA API does not return `cudaSuccess`. The thrown exception will include a description of the CUDA error code that occurred in it's `what()` message.

Example:

```
CUDA_TRY( cudaMemcpy(&dst, &src, num_bytes) );
```

## Testing

### ```column_wrapper```

The `column_wrapper<T>` class template is defined in `cudf/cpp/tests/utilities/column_wrapper.cuh` to simplify the creation and management of `gdf_column`s for the purposes of unit testing. 

`column_wrapper<T>` provides a number of constructors that allow easily constructing a `gdf_column` with the appropriate `gdf_dtype` enum set based on mapping `T` to an enum, e.g., `column_wrapper<int>` will correspond to a `gdf_column` whose `gdf_dtype` is set to `GDF_INT32`.

The simplest constructor creates an unitilized `gdf_column` of a specified type with a specified size:

```
cudf::test::column_wrapper<T>  col(size);
```

You can also construct a `gdf_column` that uses a `std::vector` to initialize the `data` and `valid` bitmask of the `gdf_column`.

```
 std::vector<T> values(size);

 std::vector<gdf_valid_type> expected_bitmask(gdf_valid_allocation_size(size), 0xFF);

 cudf::test::column_wrapper<T> const col(values, bitmask);
```

Another constructor allows passing in an initializer function that accepts a row index that will be invoked for every index `[0, size)` in the column:

```
  // This creates a gdf_column with data elements {0, 1, ..., size-1} with a valid bitmask
  // that indicates all of the values are non-null
  cudf::test::column_wrapper<T> col(size, 
      [](auto row) { return row; },
      [](auto row) { return true; });
```

To access the underlying `gdf_column` for passing into a libcudf function, the `column_wrapper::get` function can be used to provide a pointer to the underlying `gdf_column`.

```
column_wrapper<T> col(size);
gdf_column* gdf_col = col.get();
some_libcudf_function(gdf_col...);
```

