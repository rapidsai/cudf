# pylibcudf

pylibcudf is a lightweight Cython wrapper around libcudf.
It aims to provide a near-zero overhead interface to accessing libcudf in Python.
It should be possible to achieve near-native C++ performance using Cythonized code calling pylibcudf, while also allowing fairly performant usage from Python.
In addition to these requirements, pylibcudf must also integrate naturally with other Python libraries.
In other words, it should interoperate fairly transparently with standard Python containers, community protocols like `__cuda_array_interface__`, and common vocabulary types like CuPy arrays.


## General Design Principles

To satisfy the goals of pylibcudf, we impose the following set of design principles:
- Every public function or method should be `cpdef`ed. This allows it to be used in both Cython and Python code. This incurs some slight overhead over `cdef` functions, but we assume that this is acceptable because 1) the vast majority of users will be using pure Python rather than Cython, and 2) the overhead of a `cpdef` function over a `cdef` function is on the order of a nanosecond, while CUDA kernel launch overhead is on the order of a microsecond, so these function overheads should be washed out by typical usage of pylibcudf.
- Every variable used should be strongly typed and either be a primitive type (int, float, etc) or a cdef class. Any enums in C++ should be mirrored using `cpdef enum`, which will create both a C-style enum in Cython and a PEP 435-style Python enum that will automatically be used in Python.
- All typing in code should be written using Cython syntax, not PEP 484 Python typing syntax. Not only does this ensure compatibility with Cython < 3, but even with Cython 3 PEP 484 support remains incomplete as of this writing.
- All cudf code should interact only with pylibcudf, never with libcudf directly.
- All imports should be relative so that pylibcudf can be easily extracted from cudf later
  - Exception: All imports of libcudf API bindings in `cudf._lib.cpp` should use absolute imports of `cudf._lib.cpp as libcudf`. We should convert the `cpp` directory into a proper package so that it can be imported as `libcudf` in that fashion. When moving pylibcudf into a separate package, it will be renamed to `libcudf` and only the imports will need to change.
- Ideally, pylibcudf should depend on nothing other than rmm and pyarrow. This will allow it to be extracted into a a largely standalone library and used in environments where the larger dependency tree of cudf may be cumbersome.


## Relationship to libcudf

In general, the relationship between pylibcudf and libcudf can be understood in terms of two components, data structures and algorithms.

(data-structures)=

### Data Structures

Typically, every type in libcudf should have a mirror Cython `cdef` class with an attribute `self.c_obj: unique_ptr[${underlying_type}]` that owns an instance of the underlying libcudf type.
Each type should also implement a corresponding method `cdef ${cython_type} from_libcudf(${underlying_type} dt)` to enable constructing the Cython object from an underlying libcudf instance.
Depending on the nature of the type, the function may need to accept a `unique_ptr` and take ownership e.g. `cdef ${cython_type} from_libcudf(unique_ptr[${underlying_type}] obj)`.
This will typically be the case for types that own GPU data, may want to codify further.

For example, `libcudf::data_type` maps to `pylibcudf.DataType`, which looks like this (implementation omitted):
```cython

cdef class DataType:
    cdef data_type c_obj

    cpdef TypeId id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)
```

This allows pylibcudf functions to accept a typed `DataType` parameter and then trivially call underlying libcudf algorithms by accessing the argument's `c_obj`.

#### pylibcudf Tables and Columns

The primary exception to the above set of rules are libcudf's core data owning types, `cudf::table` and `cudf::column`.
libcudf uses modern C++ idioms based on smart pointers to avoid resource leaks and make code exception-safe.
To avoid passing around raw pointers, and to ensure that ownership semantics are clear, libcudf has separate `view` types corresponding to data owning types.
For example, `cudf::column` owns data, while `cudf::column_view` represents an view on a column of data and `cudf::mutable_column_view` represents a mutable view.
A `column_view` need not actually reference data owned by a `cudf::column`; any memory buffer will do.
This separation allows libcudf algorithms to clearly communicate ownership expectations and allows multiple views into the same data to coexist.

While libcudf algorithms accept views as inputs, any algorithms that allocate data must return `cudf::column` and `cudf::table` objects.
libcudf's ownership model is problematic for pylibcudf, which must be able to seamlessly interoperate with data provided by other Python libraries like PyTorch or Numba.
Therefore, pylibcudf employs the following strategy:
- pylibcudf defines the `gpumemoryview` type, which (analogous to the [Python `memoryview` type](https://docs.python.org/3/library/stdtypes.html#memoryview)) represents a view into memory owned by another object that it keeps alive using Python's standard reference counting machinery. A `gpumemoryview` is constructible from any object implementing the [CUDA Array Interface protocol](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html).
  - This type will eventually be generalized for reuse outside of pylibcudf.
- pylibcudf defines its own Table and Column classes.
  - A Table maintains Python references to the Columns it contains, so multiple Tables may share the same Column.
  - A Column consists of `gpumemoryview`s of its data buffers (which may include children for nested types) and its null mask.
- `pylibcudf.Table` and `pylibcudf.Column` provide easy access to `cudf::table_view` and `cudf::column_view` objects viewing the same columns/memory. These can be then be used when implementing any pylibcudf algorithm in terms of the underlying libcudf algorithm. Specifically, each of these classes owns an instance of the libcudf view type and provides a method `view` that may be used to access a pointer to that object to be passed to libcudf.


### Algorithms

pylibcudf algorithms should look almost exactly like libcudf algorithms.
Any libcudf function should be mirrored in pylibcudf with an identical signature and libcudf types mapped to corresponding pylibcudf types.
All calls to libcudf algorithms should perform any requisite Python preprocessing early, then release the GIL prior to calling libcudf.
For example, here is the implementation of `gather`:
```cython

cpdef Table gather(
    Table source_table,
    Column gather_map,
    OutOfBoundsPolicy bounds_policy
):
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_copying.gather(
                source_table.view(),
                gather_map.view(),
                py_policy_to_c_policy(bounds_policy)
            )
        )
    return Table.from_libcudf(move(c_result))
```

There are a couple of notable points from the snippet above:
- The object returned from libcudf is immediately converted to a pylibcudf type.
- `cudf::gather` accepts a `cudf::out_of_bounds_policy` enum parameter, which is mirrored by the `cdef `class OutOfBoundsPolicy` as mentioned in [the data structures example above](data-structures).

## Miscellaneous Notes

### Cython Scoped Enums and Casting
Cython does not support scoped enumerations.
It assumes that enums correspond to their underlying value types and will thus attempt operations that are invalid.
To fix this, many places in pylibcudf Cython code contain double casts that look like
```cython
return <cpp_type> (
    <underlying_type_t_cpp_type> py_policy
)
```
where `cpp_type` is some libcudf enum with a specified underlying type.
This double-cast will be removed when we migrate to Cython 3, which adds proper support for C++ scoped enumerations.
