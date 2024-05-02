# Supported Data Types

cuDF supports many data types supported by NumPy and Pandas, including
numeric, datetime, timedelta, categorical and string data types. We
also provide special data types for working with decimals, list-like,
and dictionary-like data.

All data types in cuDF are [nullable](missing-data).

<div class="special-table">

| Kind of data         | Data type(s)                                                                                                                             |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Signed integer       | `'int8'`, `'int16'`, `'int32'`, `'int64'`                                                                                                |
| Unsigned integer     | `'uint32'`, `'uint64'`                                                                                                                   |
| Floating-point       | `'float32'`, `'float64'`                                                                                                                 |
| Datetime             | `'datetime64[s]'`, `'datetime64[ms]'`, `'datetime64['us']`, `'datetime64[ns]'`                                                           |
| Timedelta (duration) | `'timedelta[s]'`, `'timedelta[ms]'`, `'timedelta['us']`, `'timedelta[ns]'`                                                               |
| Category             | {py:class}`~cudf.core.dtypes.CategoricalDtype`                                                                                           |
| String               | `'object'` or `'string'`                                                                                                                 |
| Decimal              | {py:class}`~cudf.core.dtypes.Decimal32Dtype`, {py:class}`~cudf.core.dtypes.Decimal64Dtype`, {py:class}`~cudf.core.dtypes.Decimal128Dtype`|
| List                 | {py:class}`~cudf.core.dtypes.ListDtype`                                                                                                  |
| Struct               | {py:class}`~cudf.core.dtypes.StructDtype`                                                                                                |

</div>

## NumPy data types

We use NumPy data types for integer, floating, datetime, timedelta,
and string data types.  Thus, just like in NumPy,
`np.dtype("float32")`, `np.float32`, and `"float32"` are all acceptable
ways to specify the `float32` data type:

```python
>>> import cudf
>>> s = cudf.Series([1, 2, 3], dtype="float32")
>>> s
0    1.0
1    2.0
2    3.0
dtype: float32
```

## A note on `object`

The data type associated with string data in cuDF is `"np.object"`.

```python
>>> import cudf
>>> s = cudf.Series(["abc", "def", "ghi"])
>>> s.dtype
dtype("object")
```

This is for compatibility with Pandas, but it can be misleading. In
both NumPy and Pandas, `"object"` is the data type associated data
composed of arbitrary Python objects (not just strings).  However,
cuDF does not support storing arbitrary Python objects.

## Decimal data types

We provide special data types for working with decimal data, namely
{py:class}`~cudf.core.dtypes.Decimal32Dtype`,
{py:class}`~cudf.core.dtypes.Decimal64Dtype`, and
{py:class}`~cudf.core.dtypes.Decimal128Dtype`.  Use these data types when you
need to store values with greater precision than allowed by floating-point
representation.

Decimal data types in cuDF are based on fixed-point representation.  A
decimal data type is composed of a _precision_ and a _scale_.  The
precision represents the total number of digits in each value of this
dtype. For example, the precision associated with the decimal value
`1.023` is `4`. The scale is the total number of digits to the right
of the decimal point. The scale associated with the value `1.023` is
3.

Each decimal data type is associated with a maximum precision:

```python
>>> cudf.Decimal32Dtype.MAX_PRECISION
9.0
>>> cudf.Decimal64Dtype.MAX_PRECISION
18.0
>>> cudf.Decimal128Dtype.MAX_PRECISION
38
```

One way to create a decimal Series is from values of type [decimal.Decimal][python-decimal].

```python
>>> from decimal import Decimal
>>> s = cudf.Series([Decimal("1.01"), Decimal("4.23"), Decimal("0.5")])
>>> s
0    1.01
1    4.23
2    0.50
dtype: decimal128
>>> s.dtype
Decimal128Dtype(precision=3, scale=2)
```

Notice the data type of the result: `1.01`, `4.23`, `0.50` can all be
represented with a precision of at least 3 and a scale of at least 2.

However, the value `1.234` needs a precision of at least 4, and a
scale of at least 3, and cannot be fully represented using this data
type:

```python
>>> s[1] = Decimal("1.234")  # raises an error
```

## Nested data types (`List` and `Struct`)

{py:class}`~cudf.core.dtypes.ListDtype` and
{py:class}`~cudf.core.dtypes.StructDtype` are special data types in cuDF for
working with list-like and dictionary-like data. These are referred to as
"nested" data types, because they enable you to store a list of lists, or a
struct of lists, or a struct of list of lists, etc.,

You can create lists and struct Series from existing Pandas Series of
lists and dictionaries respectively:

```python
>>> psr = pd.Series([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
>>> psr
0 {'a': 1, 'b': 2}
1 {'a': 3, 'b': 4}
dtype: object
>>> gsr = cudf.from_pandas(psr)
>>> gsr
0 {'a': 1, 'b': 2}
1 {'a': 3, 'b': 4}
dtype: struct
>>> gsr.dtype
StructDtype({'a': dtype('int64'), 'b': dtype('int64')})
```

Or by reading them from disk, using a [file format that supports nested data](/user_guide/io/index.md).

```python
>>> pdf = pd.DataFrame({"a": [[1, 2], [3, 4, 5], [6, 7, 8]]})
>>> pdf.to_parquet("lists.pq")
>>> gdf = cudf.read_parquet("lists.pq")
>>> gdf
           a
0     [1, 2]
1  [3, 4, 5]
2  [6, 7, 8]
>>> gdf["a"].dtype
ListDtype(int64)
```

[numpy-dtype]: https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes
[python-decimal]: https://docs.python.org/3/library/decimal.html#decimal.Decimal
