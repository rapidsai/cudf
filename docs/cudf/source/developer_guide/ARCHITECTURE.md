# cuDF Architecture

The cuDF library is a GPU-accelerated version of the [Pandas](https://pandas.pydata.org/) library.
Its primary goals are pandas API compatibility and performance.
Each of these has certain implications that inform the design of the library.

**pandas compatibility**: We endeavor to expose the same classes, methods, and free functions.
Ideally, cuDF and pandas should be interchangeable such that one library is a drop-in replacement for the other.
Note that this requirement works both ways: replacing cuDF with pandas also works as expected.
This consideration of two-way compatibility is an important guiding principle in our library's design.

**Performance**: The core of cuDF's performance comes from the underlying `libcudf` C++ library.
Most of cuDF's code is essentially a complex wrapper to provide a pandas-like interface to libcudf.
For large datasets, cuDF can usually rely on libcudf to provide speedups over CPU libraries.
However, the overhead of matching pandas's complex APIs often masks these performance benefits for smaller datasets.
Much of cuDF's architecture is designed to robustly mimic pandas APIs with minimal performance overhead.

**TODO**: Talk about interop with other libraries
**TODO**: Talk more about Arrow?

At a high level, cuDF is structured in three layers:

1. The `Frame` layer: The user-facing implementation of pandas data structures.
2. The `Column` layer: The internal Arrow-like, typed columnar data representation.
3. The `Cython` layer: A set of wrappers around the C++ libcudf library.

The emphasis of this document is describing each of these layers in turn.
Afterwards, we provide some context on other, ancillary structural components of the package.


## The Frame layer

Broadly speaking, the `Frame` layer is composed of two types of objects:
indexed table-like objects (`cudf.Series` and `cudf.DataFrame`) and index objects.
All of these classes inherit from one or both of the two base classes in this layer: `Frame` and `BaseIndex`.

### Frames
The eponymous `Frame` class contains a set of Columns and defines the methods common to all of them.
A `Frame` stores its Columns in an instance of the `ColumnAccessor` class discussed in the next section.
All user-facing classes except `RangeIndex` (see below) are subclasses of `Frame`.
The `Frame` class hierarchy is somewhat complex, reflecting the subtleties of matching the pandas API.
There are two main subclasses of interest here:

- An `IndexedFrame` is a `Frame` that has an `Index`, i.e. a `DataFrame` or `Series`.
  Due to pandas compatibility considerations, methods that _could_ be defined for any `Frame` but are not defined for
  pandas `Index` objects should be defined in `IndexedFrame`.
- A `SingleColumnFrame` is a `Frame` with a single Column.
  `Series` and every type of `Index` except `MultiIndex` is a `SingleColumnFrame`.

A `Series` is both an `IndexedFrame` and a `SingleColumnFrame`;  all other API classes inherit from one of these two.

### Indexes

The class hierarchy for Indexes in cuDF is particularly complex due to some of the constraints of the pandas API.
Before introducing that complexity, let's look at the simpler pattern structure used by the majority of classes.
At the top of this hierarchy is the `BaseIndex` class,  the parent for all indexes.
This class has no state and is largely intended to function as an abstract base class for all indexes.

```{note}
Certain functions may be implemented in `BaseIndex` if they are truly identical for all types of indexes.
However, currently most such implementations are not applicable to all subclasses and will be eventaully be removed.
```

Almost all indexes are subclasses of `GenericIndex`, which has the following class hierarchy:
`Frame`->`SingleColumnFrame`->`GenericIndex`<-`BaseIndex`.
In other words, a typical index in cuDF is a `Frame`s composed of a single Column.
For instance, integer, float, or string indexes are all examples of indexes backed by a single Column of GPU data.
While these cases represent the most common types of indexes, however, there are three notable exceptions.

The first problematic case is the `RangeIndex`.
`RangeIndex` is the only user-facing class that is not actually backed by GPU memory.
Like `pandas.RangeIndex` (or a Python `range`), this class is meant to _prevent_ memory allocations.
As a result, while `RangeIndex` is a `BaseIndex`, it is the only index type that does not also inherit from `Frame`.
Where possible, `RangeIndex` methods avoid materializing Columns to minimize device memory usage.
A subset of its functions are implemented by converting to an `Int64Index`, but we endeavor to keep these to a minimum.

The second case is the `MultiIndex`.
For obvious reasons, `MultiIndex` inherits directly from `Frame` rather than from `SingleColumnFrame`.
Like `RangeIndex`, almost all of its methods must be implemented differently from other types of indexes.

The final issue is the `Index` class itself.
In `pandas`, the `Index` class is the parent of all other types.
Although this makes sense conceptually, this inheritance pattern leads to significant headaches in concert with another
pandas decision.

The `pandas.Index` constructor is essentially a factory that will return the appropriate type of index depending on the
data type of the parameters.
For example constructing an index from a list of integers returns an `Int64Index`.
Python classes can support this type of construction via overrides of the `__new__` method.
Unfortunately, overriding `__new__` rather than (or in addition to `__init__`) is significantly more cumbersome.
Moreover, once `__new__` is overridden for a class, all its subclasses must also do the same.
This requirement makes it significantly more difficult to maintain complex inheritance trees.
Considering that multiple inheritance is used by all index classes, we wanted a way to avoid this complexity.

To solve this problem, we instead define `cudf.Index` as a subclass of `BaseIndex`.
No other index classes inherit from `cudf.Index`, nor does it define any important members.
`Index.__new__` will return the appropriate subclass of `BaseIndex` for a given input, matching pandas behavior.
A custom metaclass ensures that subclasses of `BaseIndex` appear as subclasses of `Index` to `isinstance` or `issubclass`.
This implementation simplifies all other index types while allowing `cudf.Index` to behave like `pandas.Index`.

**TODO**: Add a note about how to figure out where to add new APIs. Frame is where you should start, then if it's not
defined for Indexes, go down to IndexedFrame, then determine if it's DataFrame- or Series-only. However, if there is
shared logic between classes it may often make sense to implement a version of the function in Frame (or perhaps an
internal helper) but then override it in the child classes.


## The Column layer

The next layer in the cuDF stack is the Column layer.
The principal objects in the Column layer are the ColumnAccessor and the various Column classes.
We now consider these objects and their roles.

### ColumnAccessor

A ColumnAccessor is a dictionary-like interface to a sequence of Columns that is used to store the Columns in a Frame.
Most Frame operations are implemented as loops over ColumnAccessors that operate on their underlying Columns.
The primary purpose of the ColumnAccessor is to encapsulate pandas column selection semantics.
Columns may be selected or inserted by index or by label, and label-based selections are as flexible as pandas is.
For instance, Columns may be selected hierarchically (using tuples) or via wildcards.
ColumnAccessors also support the MultiIndex columns that can result from operations like groupbys.

### Columns

The `Column` is cuDF's core data structure and is modeled after the
[Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html).
A Column represents a sequence of values, any number of which may be "null".

Columns are typed, i.e. there are different columns for different types of data.
Thus, we have `NumericalColumn`, `StringColumn`, `DatetimeColumn`, etc.
Many `Frame` operations either behave differently or only make sense for data of a specific data type.
Those decisions should be made at the level of each `Column` subclass.
Each type of `Column` only implements methods supported by that data type.

A column is composed of the following:

- A **data type**, specifying the type of each element.
- A **data buffer** that may store the data for the column elements.
  Some column types do not have a data buffer, instead storing data in the children columns.
- A **mask buffer** whose bits represent the validity (null or not
  null) of each element. Columns whose elements are all valid may not
  have a mask buffer. Mask buffers are padded to 64 bytes.
- A tuple of **children** columns, which enable the representation of complex
  types with non-fixed width elements such as strings or lists.
- A **size** indicating the number of elements in the column.
- An integer **offset** use to represent the first element of column that is the "slice" of another column.
  The size of the column then gives the extent of the slice rather than the size of the underlying buffer.
  A column that is not a slice has an offset of 0.

As one example, a `NumericalColumn` with 1000 `int32` elements and containing nulls is composed of:

1. A data buffer of size 4000 bytes (sizeof(int32) * 1000)
2. A mask buffer of size 128 bytes (1000/8 padded to a multiple of 64
   bytes)
3. No children columns

As another example, a `StringColumn` backing the Series `['do', 'you', 'have', 'any', 'cheese?']` is composed of:

1. No data buffer
2. No mask buffer as there are no nulls in the Series
3. Two children columns:

   > - A column of UTF-8 characters
   >   `['d', 'o', 'y', 'o', 'u', 'h' ..., '?']`
   > - A column of "offsets" to the characters column (in this case,
   >   `[0, 2, 5, 9, 12, 19]`)


### Data types

Data types, or `dtypes`, are extensions of the
[data type objects introduced by numpy](https://numpy.org/doc/stable/reference/arrays.dtypes.html).
cuDF supports most standard data types, with the notable exception of the arbitrary `object` dtype.
Efficient GPU algorithms generally require knowledge of data layouts, making arbitrary objects infeasible to handle.

For this purpose, cuDF in fact defines certain additional `dtypes` to handle common uses cases of the `object` dtype.
All cuDF `dtypes` subclass the pandas `ExtensionDtype`.
The following are the list of cuDF's extension `dtypes` along with a description of elements of that type:
- `ListDtype`: Lists where each element in every list in a Column is of the same type.
- `StructDtype`: Dicts where a given key always maps to values of the same type
- `CategoricalDtype`: Analogous to the pandas categorical dtype except that the categories are stored in device memory.
- `DecimalDtype`: Fixed-point numbers
- `IntervalDtype`: Intervals

Note that there is a many-to-one mapping between data type and `Column` class.
For instance, all numerical types (floats and ints of different widths) are all managed using `NumericalColumn`.


### Buffer

Columns in cuDF do not directly own their memory; instead, memory ownership is handled at the level of the `Buffer` object.
A `Buffer` represents a device memory allocation that it _may or may not_ own.
A `Buffer` constructed from a preexisting device memory allocation (such as a CuPy array) will simply view that memory.
Conversely, a `Buffer` constructed from a host object will allocate new device memory and copy in the data.
cuDF uses the [RMM](https://github.com/rapidsai/rmm) library for allocating device memory.
You can read more about device memory allocation with RMM [here](https://github.com/rapidsai/rmm#devicebuffers).

```{note}
cuDF needs to interoperate with a wide range of Python libraries, many of which allocate device memory.
The ownership behavior described above is designed to minimize new memory allocations for externally-owned memory.
Developers familiar with cuDF-`libcudf` interoperability will recognize a discrepancy in this ownership model.
cuDF's employs `libcudf` under the hood, but libcudf objects aren't Python objects, so something must own their memory.

The key is to recognize that `libcudf` offers both owning (e.g. `column`) and non-owning (e.g. `column_view`) objects.
All `libcudf` algorithms accept views as parameters while returning (new) owning objects.
When calling `libcudf` APIs, cuDF Python construct views from cudf `Buffers`.
When owning objects are returned, cuDF has an rmm object take ownership of that memory and stores that in a `Buffer`.
The result is that the `Buffer` always owns memory allocated by `libcudf`.
```

## The Cython layer

The lowest level of cuDF is its interaction with `libcudf` via Cython.
Most algorithms in cudf follow a similar pattern.
The `Frame` layer processes inputs and calls a `Column` method.
That method in turn does some additional processing before finally calling a Cython function.
The result is then passed back up through the layers, undergoing postprocessing as needed.

The Cython layer itself is largely composed of two parts: C++ bindings and Cython wrappers.
We use Cython to expose C++ functionality to Python.
This code essentially consists of copying libcudf header files into new files with a slightly different format.
Since these bindings are only accessible from Cython, we write Cython wrappers that can be called from pure Python code.
These wrappers translate cuDF objects into their `libcudf` equivalents and then invoke `libcudf` functions.

We endeavor to make these wrappers as thin as possible.
By the time code reaches this layer, all questions of pandas compatibility should already have been addressed.


## Misc

### Mixins

### Scalar
