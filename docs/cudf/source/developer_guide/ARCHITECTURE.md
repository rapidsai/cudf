# cuDF Architecture

The cuDF library is a GPU-accelerated, [Pandas-like](https://pandas.pydata.org/) DataFrame library.
pandas APIs provide users a greate deal of power and flexibility, and we aim to match that.
As a result, a key design challenge for cuDF is finding the simplest, most performant approaches to mimic pandas APIs.

At a high level, cuDF is structured in three layers, each of which serves a distinct purpose in this regard:

1. The Frame layer: The user-facing implementation of pandas-like data structures.
2. The Column layer: The core internal data structures used to bridge the gap to our lower-level implementations.
3. The Cython layer: The wrappers around the fast C++ `libcudf` library.

In this document we will review each of these layers, their roles, and the requisite tradeoffs.


## The Frame layer

Broadly speaking, the `Frame` layer is composed of two types of objects: indexed tables and indexes.
The mapping between these types and cuDF data types is not obvious, however.
To ease our way into understanding why, let's first take a birds-eye view of the Frame layer.

All classes in this layer inherit from one or both of the two base classes in this layer: `Frame` and `BaseIndex`.
The eponymous `Frame` class is, at its core, a simple tabular data structure composed of columnar data.
Some types of `Frame` contain indexes; in particular, any `DataFrame` or `Series` has an index.
However, as a general container of columnar data, `Frame` is also the parent class for most types of index.

`BaseIndex`, meanwhile, is essentially an abstract base class encoding the `pandas.Index` API.
Various subclasses of `BaseIndex` implement this API in specific ways depending on their underlying data.
For example, `RangeIndex` avoids actually materializing a column, while a `MultiIndex` contains _multiple_ columns.
Most other index classes consist of a single column of a given type, e.g. strings or datetimes.
As a result, using a single abstract parent provides the flexibility we need to support these different types.

With those preliminaries out of the way, let's dive in a little bit deeper.

### Frames

`Frame` exposes numerous methods common to all pandas data structures.
Any methods that have the same API across `Series`, `DataFrame`, and `Index` should be defined here.
Additionally any (internal) methods that could be used to share code between those classes may also be defined here.

The primary internal subclass of `Frame` is `IndexedFrame`, a `Frame` with an index.
An `IndexedFrame` represents the first type of object mentioned above: indexed tables.
In particular, `IndexedFrame` is a parent class for `DataFrame` and `Series`.
Any pandas methods that are defined for those two classes should be defined here.

The second internal subclass of `Frame` is `SingleColumnFrame`.
As you may surmise, it is a `Frame` with a single column of data.
This class is a parent for most types of indexes as well as `Series` (note the diamond inheritance pattern here).
While `IndexedFrame` provides a large amount of functionality, this class is much simpler.
It adds some simple APIs provided by all 1D pandas objects, and it flattens outputs where needed.

### Indexes

While we've highlighted some exceptional cases of Indexes before, let's start with the base cases here first.
`BaseIndex` is generally intended to be a true abstract class, i.e. it should contain no implementations.
Functions may be implemented in `BaseIndex` if they are truly identical for all types of indexes.
However, currently most such implementations are not applicable to all subclasses and will be eventually be removed.

Almost all indexes are subclasses of `GenericIndex`, a single-columned index with the class hierarchy:
```python
class GenericIndex(SingleColumnFrame, BaseIndex)
```
`Frame`->`SingleColumnFrame`->`GenericIndex`<-`BaseIndex`.
Integer, float, or string indexes are all composed of a single column of data.
Most `GenericIndex` methods are inherited from `Frame`, saving us the trouble of rewriting them.

We now consider the three main exceptions to this model:

- A `RangeIndex` is not backed by a column of data, so it inherits directly from `BaseIndex` alone.
  Wherever possible, its methods have special implementations designed to avoid materializing columns.
  Where such an implementation is infeasible, we fall back to converting it to an integer index first instead.
- A `MultiIndex` is backed by _multiple_ columns of data.
  Therefore, its inheritance hierarchy looks like `class MultiIndex(Frame, BaseIndex)`.
  Some of its more `Frame`-like methods may be inherited,
  but many others must be reimplemented since in many cases a `MultiIndex` is not expected to behave like a `Frame`.
- Just like in pandas, `Index` itself can never be instantiated.
  `pandas.Index` is the parent class for indexes,
  but its constructor returns an appropriate subclass depending on the input data type and shape.
  Unfortunately, mimicking this behavior requires overriding `__new__`,
  which in turn makes shared initialization across inheritance trees much more cumbersome to manage.
  To enable sharing constructor logic across different index classes,
  we instead define `BaseIndex` as the parent class of all indexes.
  `Index` inherits from `BaseIndex`, but it masquerades as a `BaseIndex` to match pandas.
  This class should contain no implementations since it is simply a factory for other indexes.


## The Column layer

The next layer in the cuDF stack is the Column layer.
This layer forms the glue between pandas-like APIs and our underlying data layouts.
Under the hood, cuDF is built around the [Apache Arrow Format](https://arrow.apache.org).
This data format is both conducive to high-performance algorithms and suitable for data interchange between libraries.

The principal objects in the Column layer are the `ColumnAccessor` and the various `Column` classes.
The `Column` is cuDF's core data structure that represents a single column of data of a specific data type.
A `ColumnAccessor` is a dictionary-like interface to a sequence of `Column`s.
A `Frame` owns a `ColumnAccessor`, and most of its operations are implemented as loops over that object's `Column`s.

### ColumnAccessor

The primary purpose of the `ColumnAccessor` is to encapsulate pandas column selection semantics.
Columns may be selected or inserted by index or by label, and label-based selections are as flexible as pandas is.
For instance, Columns may be selected hierarchically (using tuples) or via wildcards.
`ColumnAccessors` also support the `MultiIndex` columns that can result from operations like groupbys.

### Columns

The parent `Column` class is implemented in Cython to support interchange with C++ (more on that later).
However, the bulk of the `Column`'s functionality is embodied by the `ColumnBase` subclass.
`ColumnBase` provides many standard methods, while others only make sense for data of a specific type.
As a result, we have various subclasses of `ColumnBase` like `NumericalColumn`, `StringColumn`, and `DatetimeColumn`.
Most dtype-specific decisions should be handled at the level of a specific `Column` subclass.
Each type of `Column` only implements methods supported by that data type.

A `Column` is composed of the following:

- A **data type**, specifying the type of each element.
- A **data buffer** that may store the data for the column elements.
  Some column types do not have a data buffer, instead storing data in the children columns.
- A **mask buffer** whose bits represent the validity (null or not null) of each element.
  Nullability is a core concept in the Arrow data model.
  Columns whose elements are all valid may not have a mask buffer.
  Mask buffers are padded to 64 bytes.
- Its **children**, a tuple of columns used to represent complex types such as structs or lists.
- A **size** indicating the number of elements in the column.
- An integer **offset** use to represent the first element of column that is the "slice" of another column.
  The size of the column then gives the extent of the slice rather than the size of the underlying buffer.
  A column that is not a slice has an offset of 0.

More information about these fields can be found in the documentation of the
[Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html),
which is what the cuDF `Column` is based on.

As one example, a `NumericalColumn` with 1000 `int32` elements and containing nulls is composed of:

1. A data buffer of size 4000 bytes (sizeof(int32) * 1000)
2. A mask buffer of size 128 bytes (1000/8 padded to a multiple of 64
   bytes)
3. No children columns

As another example, a `StringColumn` backing the Series `['do', 'you', 'have', 'any', 'cheese?']` is composed of:

1. No data buffer
2. No mask buffer as there are no nulls in the Series
3. Two children columns:

   - A column of UTF-8 characters
     `['d', 'o', 'y', 'o', 'u', 'h' ..., '?']`
   - A column of "offsets" to the characters column (in this case,
     `[0, 2, 5, 9, 12, 19]`)


### Data types

cuDF uses [dtypes](https://numpy.org/doc/stable/reference/arrays.dtypes.html) to represent different types of data.
Since efficient GPU algorithms require preexisting knowledge of data layouts,
cuDF does not support the arbitrary `object` dtype, but instead defines a few custom types for common use-cases:
- `ListDtype`: Lists where each element in every list in a Column is of the same type
- `StructDtype`: Dicts where a given key always maps to values of the same type
- `CategoricalDtype`: Analogous to the pandas categorical dtype except that the categories are stored in device memory
- `DecimalDtype`: Fixed-point numbers
- `IntervalDtype`: Intervals

Note that there is a many-to-one mapping between data types and `Column` classes.
For instance, all numerical types (floats and ints of different widths) are all managed using `NumericalColumn`.


### Buffer

Although a `Column` represents an Arrow-compliant data structure, it does not directly handle memory mangaement.
That job is delegated to the `Buffer` class, which represents a device memory allocation that it _may or may not_ own.
A `Buffer` constructed from a preexisting device memory allocation (such as a CuPy array) will view that memory.
Conversely, a `Buffer` constructed from a host object will allocate new device memory and copy in the data.
cuDF uses the [RMM](https://github.com/rapidsai/rmm) library for allocating device memory.
You can read more about device memory allocation with RMM [here](https://github.com/rapidsai/rmm#devicebuffers).

```{note}
cuDF needs to interoperate with a wide range of Python libraries, many of which allocate device memory.
The ownership behavior described above is designed to minimize new memory allocations for externally-owned memory.
cuDF's use of `libcudf`, however, represents a slight break from the paradigm employed above:
since `libcudf` objects are not Python objects, we need something to manage their lifetimes from within cuDF.

The key is to recognize that `libcudf` offers both owning (e.g. `column`) and non-owning (e.g. `column_view`) objects.
All `libcudf` algorithms accept views as parameters while returning (new) owning objects.
When calling `libcudf` APIs, cuDF Python constructs views from cuDF `Buffers`.
When owning objects are returned, cuDF has an rmm object take ownership of that memory and stores that in a `Buffer`.
The result is that all memory allocated by `libcudf` inside cuDF eventually has ownership transferred to a `Buffer`.
```

## The Cython layer

The lowest level of cuDF is its interaction with `libcudf` via Cython.
Most algorithms in cuDF follow a similar pattern.
The `Frame` layer processes inputs and calls a `Column` method.
That method in turn does some additional processing before finally calling a Cython function.
The result is then passed back up through the layers, undergoing postprocessing as needed.

The Cython layer itself is largely composed of two parts: C++ bindings and Cython wrappers.
In the first case, we expose `libcudf`'s C++ functionality to Cython by writing 
[Cython `.pxd` (declaration) files](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html)
that consist of Cython declarations equivalent to the contents of corresponding C++ header files.
These `pxd` files make the contents of those C++ headers callable from Cython, but not pure Python.
We then wrap these functions with wrappers written in Cython, which in turn _can_ be used in Python code.
These wrappers translate cuDF objects into their `libcudf` equivalents and then invoke `libcudf` functions.

We endeavor to make these wrappers as thin as possible.
By the time code reaches this layer, all questions of pandas compatibility should already have been addressed.
These functions should be as close to trivial wrappers around `libcudf` APIs as possible.
