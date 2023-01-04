# Comparison of cuDF and Pandas

cuDF is a DataFrame library that closely matches the Pandas API, but
it is *not* a full drop-in replacement for Pandas.  There are some
differences between cuDF and Pandas, both in terms of API and
behaviour.  This page documents the similarities and differences
between cuDF and Pandas.

## Supported operations

cuDF supports many of the same data structures and operations as
Pandas.  This includes `Series`, `DataFrame`, `Index` and
operations on them such as unary and binary operations, indexing,
filtering, concatenating, joining, groupby and window operations -
among many others.

The best way to check if we support a particular Pandas API is to search
our [API docs](/api_docs/index).

## Data types

cuDF supports many of the commonly-used data types in Pandas,
including numeric, datetime, timestamp, string, and categorical data
types.  In addition, we support special data types for decimal, list,
and "struct" values.  See the section on [Data Types](data-types) for
details.

Note that we do not support custom data types like Pandas'
`ExtensionDtype`.

## Null (or "missing") values

Unlike Pandas, *all* data types in cuDF are nullable,
meaning they can contain missing values (represented by `cudf.NA`).

```{code} python
>>> s = cudf.Series([1, 2, cudf.NA])
>>> s
0       1
1       2
2    <NA>
dtype: int64
```

Nulls are not coerced to `NaN` in any situation;
compare the behavior of cuDF with Pandas below:

```{code} python
>>> s = cudf.Series([1, 2, cudf.NA], dtype="category")
>>> s
0       1
1       2
2    <NA>
dtype: category
Categories (2, int64): [1, 2]

>>> s = pd.Series([1, 2, pd.NA], dtype="category")
>>> s
0      1
1      2
2    NaN
dtype: category
Categories (2, int64): [1, 2]
```

See the docs on [missing data](missing-data) for details.

## Iteration

Iterating over a cuDF `Series`, `DataFrame` or `Index` is not
supported. This is because iterating over data that resides on the GPU
will yield *extremely* poor performance, as GPUs are optimized for
highly parallel operations rather than sequential operations.

In the vast majority of cases, it is possible to avoid iteration and
use an existing function or method to accomplish the same task. If you
absolutely must iterate, copy the data from GPU to CPU by using
`.to_arrow()` or `.to_pandas()`, then copy the result back to GPU
using `.from_arrow()` or `.from_pandas()`.

## Result ordering

By default, `join` (or `merge`) and `groupby` operations in cuDF
do *not* guarantee output ordering.
Compare the results obtained from Pandas and cuDF below:

```{code} python
 >>> import cupy as cp
 >>> df = cudf.DataFrame({'a': cp.random.randint(0, 1000, 1000), 'b': range(1000)})
 >>> df.groupby("a").mean().head()
          b
 a
 742  694.5
 29   840.0
 459  525.5
 442  363.0
 666    7.0
 >>> df.to_pandas().groupby("a").mean().head()
          b
 a
 2   643.75
 6    48.00
 7   631.00
 9   906.00
 10  640.00
```

To match Pandas behavior, you must explicitly pass `sort=True`:

```{code} python
>>> df.to_pandas().groupby("a", sort=True).mean().head()
         b
a
2   643.75
6    48.00
7   631.00
9   906.00
10  640.00
```

## Floating-point computation

cuDF leverages GPUs to execute operations in parallel.  This means the
order of operations is not always deterministic.  This impacts the
determinism of floating-point operations because floating-point
arithmetic is non-associative, that is, `a + b` is not equal to `b + a`.

For example, `s.sum()` is not guaranteed to produce identical results
to Pandas nor produce identical results from run to run, when `s` is a
Series of floats.  If you need to compare floating point results, you
should typically do so using the functions provided in the
[`cudf.testing`](/api_docs/general_utilities#testing-functions)
module, which allow you to compare values up to a desired precision.

## Column names

Unlike Pandas, cuDF does not support duplicate column names.
It is best to use unique strings for column names.

## No true `"object"` data type

In Pandas and NumPy, the `"object"` data type is used for
collections of arbitrary Python objects.  For example, in Pandas you
can do the following:

```{code} python
>>> import pandas as pd
>>> s = pd.Series(["a", 1, [1, 2, 3]])
0            a
1            1
2    [1, 2, 3]
dtype: object
```

For compatibility with Pandas, cuDF reports the data type for strings
as `"object"`, but we do *not* support storing or operating on
collections of arbitrary Python objects.

## `.apply()` function limitations

The `.apply()` function in Pandas accepts a user-defined function
(UDF) that can include arbitrary operations that are applied to each
value of a `Series`, `DataFrame`, or in the case of a groupby,
each group.  cuDF also supports `.apply()`, but it relies on Numba to
JIT compile the UDF and execute it on the GPU. This can be extremely
fast, but imposes a few limitations on what operations are allowed in
the UDF. See the docs on [UDFs](guide-to-udfs) for details.
