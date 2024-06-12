# Breaking changes for pandas 2 in cuDF 24.04+

In release 24.04 and later, cuDF requires pandas 2, following the announcement in [RAPIDS Support Notice 36](https://docs.rapids.ai/notices/rsn0036/).
Migrating to pandas 2 comes with a number of API and behavior changes, documented below.
The changes to support pandas 2 affect both `cudf` and `cudf.pandas` (cuDF pandas accelerator mode).
For more details, refer to the [pandas 2.0 changelog](https://pandas.pydata.org/docs/whatsnew/index.html#version-2-0).

## Removed `DataFrame.append` & `Series.append`, use `cudf.concat` instead.

`DataFrame.append` & `Series.append` deprecations are enforced by removing these two APIs. Instead, please use `cudf.concat`.

Old behavior:
```python

In [37]: s = cudf.Series([1, 2, 3])

In [38]: p = cudf.Series([10, 20, 30])

In [39]: s.append(p)
Out[39]:
0     1
1     2
2     3
0    10
1    20
2    30
dtype: int64
```

New behavior:
```python
In [40]: cudf.concat([s, p])
Out[40]:
0     1
1     2
2     3
0    10
1    20
2    30
dtype: int64
```


## Removed various numeric `Index` sub-classes, use `cudf.Index`

`Float32Index`, `Float64Index`, `GenericIndex`, `Int8Index`, `Int16Index`, `Int32Index`, `Int64Index`, `StringIndex`, `UInt8Index`, `UInt16Index`, `UInt32Index`, `UInt64Index` have all been removed, use `cudf.Index` directly with a `dtype` to construct the index instead.

Old behavior:
```python
In [35]: cudf.Int8Index([0, 1, 2])
Out[35]: Int8Index([0, 1, 2], dtype='int8')
```

New behavior:
```python
In [36]: cudf.Index([0, 1, 2], dtype='int8')
Out[36]: Index([0, 1, 2], dtype='int8')
```


## Change in bitwise operation results


Bitwise operations between two objects with different indexes will now not result in boolean results.


Old behavior:

```python
In [1]: import cudf

In [2]: import numpy as np

In [3]: s = cudf.Series([1, 2, 3])

In [4]: p = cudf.Series([10, 11, 12], index=[2, 1, 10])

In [5]: np.bitwise_or(s, p)
Out[5]:
0      True
1      True
2      True
10    False
dtype: bool
```

New behavior:
```python
In [5]: np.bitwise_or(s, p)
Out[5]:
0     <NA>
1       11
2       11
10    <NA>
dtype: int64
```


## ufuncs will perform re-indexing

Performing a numpy ufunc operation on two objects with mismatching index will result in re-indexing:

Old behavior:

```python
In [1]: import cudf

In [2]: df = cudf.DataFrame({"a": [1, 2, 3]}, index=[0, 2, 3])

In [3]: df1 = cudf.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 3])

In [4]: import numpy as np

In [6]: np.add(df, df1)
Out[6]:
   a
0  2
2  4
3  6
```


New behavior:

```python
In [6]: np.add(df, df1)
Out[6]:
       a
0   <NA>
2   <NA>
3      6
10  <NA>
20  <NA>
```


## `DataFrame` vs `Series` comparisons need to have matching index

Going forward any comparison between `DataFrame` & `Series` objects will need to have matching axes, i.e., the column names of `DataFrame` should match index of `Series`:

Old behavior:
```python
In [1]: import cudf

In [2]: df = cudf.DataFrame({'a':range(0, 5), 'b':range(10, 15)})

In [3]: df
Out[3]:
   a   b
0  0  10
1  1  11
2  2  12
3  3  13
4  4  14

In [4]: s = cudf.Series([1, 2, 3])

In [6]: df == s
Out[6]:
       a      b      0      1      2
0  False  False  False  False  False
1  False  False  False  False  False
2  False  False  False  False  False
3  False  False  False  False  False
4  False  False  False  False  False
```

New behavior:
```python
In [5]: df == s

ValueError: Can only compare DataFrame & Series objects whose columns & index are same respectively, please reindex.

In [8]: s = cudf.Series([1, 2], index=['a', 'b'])
# Create a series with matching Index to that of `df.columns` and then compare.

In [9]: df == s
Out[9]:
       a      b
0  False  False
1   True  False
2  False  False
3  False  False
4  False  False
```


## Series.rank


`Series.rank` will now throw an error for non-numeric data when `numeric_only=True` is passed:

Old behavior:

```python
In [4]: s = cudf.Series(["a", "b", "c"])
   ...: s.rank(numeric_only=True)
Out[4]: Series([], dtype: float64)
```

New behavior:

```python
In [4]: s = cudf.Series(["a", "b", "c"])
   ...: s.rank(numeric_only=True)
TypeError: Series.rank does not allow numeric_only=True with non-numeric dtype.
```



## Value counts sets the results name to `count`/`proportion`


In past versions, when running `Series.value_counts()`, the result would inherit the original object's name, and the result index would be nameless. This would cause confusion when resetting the index, and the column names would not correspond with the column values. Now, the result name will be `'count'` (or `'proportion'` if `normalize=True`).

Old behavior:
```python
In [3]: cudf.Series(['quetzal', 'quetzal', 'elk'], name='animal').value_counts()

Out[3]:
quetzal    2
elk        1
Name: animal, dtype: int64
```

New behavior:
```python
In [3]: pd.Series(['quetzal', 'quetzal', 'elk'], name='animal').value_counts()

Out[3]:
animal
quetzal    2
elk        1
Name: count, dtype: int64
```


## `DataFrame.describe` will include datetime data by default

Previously by default (i.e., `datetime_is_numeric=False`) `describe` would not return datetime data. Now this parameter is inoperative will always include datetime columns.

Old behavior:
```python
In [4]: df = cudf.DataFrame(
   ...:             {
   ...:                 "int_data": [1, 2, 3],
   ...:                 "str_data": ["hello", "world", "hello"],
   ...:                 "float_data": [0.3234, 0.23432, 0.0],
   ...:                 "timedelta_data": cudf.Series(
   ...:                     [1, 2, 1], dtype="timedelta64[ns]"
   ...:                 ),
   ...:                 "datetime_data": cudf.Series(
   ...:                     [1, 2, 1], dtype="datetime64[ns]"
   ...:                 ),
   ...:             }
   ...:         )
   ...:

In [5]: df
Out[5]:
   int_data str_data  float_data            timedelta_data                 datetime_data
0         1    hello     0.32340 0 days 00:00:00.000000001 1970-01-01 00:00:00.000000001
1         2    world     0.23432 0 days 00:00:00.000000002 1970-01-01 00:00:00.000000002
2         3    hello     0.00000 0 days 00:00:00.000000001 1970-01-01 00:00:00.000000001

In [6]: df.describe()
Out[6]:
       int_data  float_data             timedelta_data
count       3.0    3.000000                          3
mean        2.0    0.185907  0 days 00:00:00.000000001
std         1.0    0.167047            0 days 00:00:00
min         1.0    0.000000  0 days 00:00:00.000000001
25%         1.5    0.117160  0 days 00:00:00.000000001
50%         2.0    0.234320  0 days 00:00:00.000000001
75%         2.5    0.278860  0 days 00:00:00.000000001
max         3.0    0.323400  0 days 00:00:00.000000002
```

New behavior:
```python
In [6]: df.describe()
Out[6]:
       int_data  float_data             timedelta_data                  datetime_data
count       3.0    3.000000                          3                              3
mean        2.0    0.185907  0 days 00:00:00.000000001  1970-01-01 00:00:00.000000001
min         1.0    0.000000  0 days 00:00:00.000000001  1970-01-01 00:00:00.000000001
25%         1.5    0.117160  0 days 00:00:00.000000001  1970-01-01 00:00:00.000000001
50%         2.0    0.234320  0 days 00:00:00.000000001  1970-01-01 00:00:00.000000001
75%         2.5    0.278860  0 days 00:00:00.000000001  1970-01-01 00:00:00.000000001
max         3.0    0.323400  0 days 00:00:00.000000002  1970-01-01 00:00:00.000000002
std         1.0    0.167047            0 days 00:00:00                           <NA>
```



## Converting a datetime string with `Z` to timezone-naive dtype is not allowed.

Previously when a date that had `Z` at the trailing end was allowed to be type-casted to `datetime64` type, now that will raise an error.

Old behavior:
```python
In [11]: s = cudf.Series(np.datetime_as_string(np.arange("2002-10-27T04:30", 10 * 60, 1, dtype="M8[m]"),timezone="UTC"))

In [12]: s
Out[12]:
0      2002-10-27T04:30Z
1      2002-10-27T04:31Z
2      2002-10-27T04:32Z
3      2002-10-27T04:33Z
4      2002-10-27T04:34Z
             ...
595    2002-10-27T14:25Z
596    2002-10-27T14:26Z
597    2002-10-27T14:27Z
598    2002-10-27T14:28Z
599    2002-10-27T14:29Z
Length: 600, dtype: object

In [13]: s.astype('datetime64[ns]')
Out[13]:
0     2002-10-27 04:30:00
1     2002-10-27 04:31:00
2     2002-10-27 04:32:00
3     2002-10-27 04:33:00
4     2002-10-27 04:34:00
              ...
595   2002-10-27 14:25:00
596   2002-10-27 14:26:00
597   2002-10-27 14:27:00
598   2002-10-27 14:28:00
599   2002-10-27 14:29:00
Length: 600, dtype: datetime64[ns]
```

New behavior:
```python
In [13]: s.astype('datetime64[ns]')

*** NotImplementedError: cuDF does not yet support timezone-aware datetimes casting
```


## `Datetime` & `Timedelta` reduction operations will preserve their time resolutions.


Previously reduction operations on `datetime64` & `timedelta64` types used to result in lower-resolution results.
Now the original resolution is preserved:

Old behavior:
```python
In [14]: sr = cudf.Series([10, None, 100, None, None], dtype='datetime64[us]')

In [15]: sr
Out[15]:
0    1970-01-01 00:00:00.000010
1                          <NA>
2    1970-01-01 00:00:00.000100
3                          <NA>
4                          <NA>
dtype: datetime64[us]

In [16]: sr.std()
Out[16]: Timedelta('0 days 00:00:00.000063639')
```

New behavior:
```python
In [16]: sr.std()
Out[16]: Timedelta('0 days 00:00:00.000063')
```


## `get_dummies` default return type is changed from `int8` to `bool`

The default return values of `get_dummies` will be `boolean` instead of `int8`

Old behavior:
```python
In [2]: s = cudf.Series([1, 2, 10, 11, None])

In [6]: cudf.get_dummies(s)
Out[6]:
   1   2   10  11
0   1   0   0   0
1   0   1   0   0
2   0   0   1   0
3   0   0   0   1
4   0   0   0   0
```

New behavior:
```python
In [3]: cudf.get_dummies(s)
Out[3]:
      1      2      10     11
0   True  False  False  False
1  False   True  False  False
2  False  False   True  False
3  False  False  False   True
4  False  False  False  False
```

## `reset_index` will name columns as `None` when `name=None`

`reset_index` used to name columns as `0` or `self.name` if `name=None`. Now, passing `name=None` will name the column as `None` exactly.

Old behavior:
```python
In [2]: s = cudf.Series([1, 2, 3])

In [4]: s.reset_index(name=None)
Out[4]:
   index  0
0      0  1
1      1  2
2      2  3
```

New behavior:
```python
In [7]: s.reset_index(name=None)
Out[7]:
   index  None
0      0     1
1      1     2
2      2     3
```

## Fixed an issue where duration components were being incorrectly calculated

Old behavior:

```python
In [18]: sr = cudf.Series([136457654736252, 134736784364431, 245345345545332, 223432411, 2343241, 3634548734, 23234], dtype='timedelta64[ms]')

In [19]: sr
Out[19]:
0    1579371 days 00:05:36.252
1    1559453 days 12:32:44.431
2    2839645 days 04:52:25.332
3          2 days 14:03:52.411
4          0 days 00:39:03.241
5         42 days 01:35:48.734
6          0 days 00:00:23.234
dtype: timedelta64[ms]

In [21]: sr.dt.components
Out[21]:
    days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
0  84843      3        3       40           285           138          688
1  64925     15       30       48           464           138          688
2  64093     10       23        7           107           828          992
3      2     14        3       52           411             0            0
4      0      0       39        3           241             0            0
5     42      1       35       48           734             0            0
6      0      0        0       23           234             0            0
```

New behavior:

```python
In [21]: sr.dt.components
Out[21]:
      days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
0  1579371      0        5       36           252             0            0
1  1559453     12       32       44           431             0            0
2  2839645      4       52       25           332             0            0
3        2     14        3       52           411             0            0
4        0      0       39        3           241             0            0
5       42      1       35       48           734             0            0
6        0      0        0       23           234             0            0
```


## `fillna` on `datetime`/`timedelta` with a lower-resolution scalar will now type-cast the series

Previously, when `fillna` was performed with a higher-resolution scalar than the series, the resulting resolution would have been cast to the higher resolution. Now the original resolution is preserved.

Old behavior:
```python
In [22]: sr = cudf.Series([1000000, 200000, None], dtype='timedelta64[s]')

In [23]: sr
Out[23]:
0    11 days 13:46:40
1     2 days 07:33:20
2                <NA>
dtype: timedelta64[s]

In [24]: sr.fillna(np.timedelta64(1,'ms'))
Out[24]:
0       11 days 13:46:40
1        2 days 07:33:20
2    0 days 00:00:00.001
dtype: timedelta64[ms]
```

New behavior:
```python
In [24]: sr.fillna(np.timedelta64(1,'ms'))
Out[24]:
0    11 days 13:46:40
1     2 days 07:33:20
2     0 days 00:00:00
dtype: timedelta64[s]
```

## `Groupby.nth` & `Groupby.dtypes` will have the grouped column in result

Previously, `Groupby.nth` & `Groupby.dtypes` would set the grouped columns as `Index` object. Now the new behavior will actually preserve the original objects `Index` and return the grouped columns too as part of the result.

Old behavior:
```python
In [31]: df = cudf.DataFrame(
    ...:         {
    ...:             "a": [1, 1, 1, 2, 3],
    ...:             "b": [1, 2, 2, 2, 1],
    ...:             "c": [1, 2, None, 4, 5],
    ...:             "d": ["a", "b", "c", "d", "e"],
    ...:         }
    ...:     )
    ...:

In [32]: df
Out[32]:
   a  b     c  d
0  1  1     1  a
1  1  2     2  b
2  1  2  <NA>  c
3  2  2     4  d
4  3  1     5  e

In [33]: df.groupby('a').nth(1)
Out[33]:
   b  c  d
a
1  2  2  b

In [34]: df.groupby('a').dtypes
Out[34]:
       b      c       d
a
1  int64  int64  object
2  int64  int64  object
3  int64  int64  object
```

New behavior:
```python
In [33]: df.groupby('a').nth(1)
Out[33]:
   a  b    c  d
1  1  2  2.0  b

In [34]: df.groupby('a').dtypes
Out[34]:
       a      b      c       d
a
1  int64  int64  int64  object
2  int64  int64  int64  object
3  int64  int64  int64  object
```
