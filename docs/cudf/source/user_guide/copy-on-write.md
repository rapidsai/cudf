(copy-on-write-user-doc)=

# Copy-on-write

Copy-on-write reduces GPU memory usage when copies(`.copy(deep=False)`) of a column
are made.

|                     | Copy-on-Write enabled                                                                                                                                                                                          | Copy-on-Write disabled (default)                                                                               |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `.copy(deep=True)`  | A true copy is made and changes don't propagate to the original object.                                                                                                                            | A true copy is made and changes don't propagate to the original object.                  |
| `.copy(deep=False)` | Memory is shared between the two objects and but any write operation on one object will trigger a true physical copy before the write is performed. Hence changes will not propagate to the original object. | Memory is shared between the two objects and changes performed on one will propagate to the other object. |

## How to enable it

i. Use `cudf.set_option`:

```python
>>> import cudf
>>> cudf.set_option("copy_on_write", True)
```

ii. Set the environment variable ``CUDF_COPY_ON_WRITE`` to ``1`` prior to the
launch of the Python interpreter:

```bash
export CUDF_COPY_ON_WRITE="1" python -c "import cudf"
```


## Making copies

There are no additional changes required in the code to make use of copy-on-write.

```python
>>> series = cudf.Series([1, 2, 3, 4])
```

Performing a shallow copy will create a new Series object pointing to the
same underlying device memory:

```python
>>> copied_series = series.copy(deep=False)
>>> series
0    1
1    2
2    3
3    4
dtype: int64
>>> copied_series
0    1
1    2
2    3
3    4
dtype: int64
```

When a write operation is performed on either ``series`` or
``copied_series``, a true physical copy of the data is created:

```python
>>> series[0:2] = 10
>>> series
0    10
1    10
2     3
3     4
dtype: int64
>>> copied_series
0    1
1    2
2    3
3    4
dtype: int64
```


## Notes

When copy-on-write is enabled, there is no concept of views. i.e., modifying any view created inside cudf will not actually not modify
the original object it was viewing and thus a separate copy is created and then modified.

## Advantages

1. With the concept of views going away, every object is a copy of it's original object. This will bring consistency across operations and cudf closer to parity with
pandas. Following is one of the inconsistency:

```python

>>> import pandas as pd
>>> s = pd.Series([1, 2, 3, 4, 5])
>>> s1 = s[0:2]
>>> s1[0] = 10
>>> s1
0    10
1     2
dtype: int64
>>> s
0    10
1     2
2     3
3     4
4     5
dtype: int64

>>> import cudf
>>> s = cudf.Series([1, 2, 3, 4, 5])
>>> s1 = s[0:2]
>>> s1[0] = 10
>>> s1
0    10
1     2
>>> s
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

The above inconsistency is solved when Copy-on-write is enabled:

```python
>>> import pandas as pd
>>> pd.set_option("mode.copy_on_write", True)
>>> s = pd.Series([1, 2, 3, 4, 5])
>>> s1 = s[0:2]
>>> s1[0] = 10
>>> s1
0    10
1     2
dtype: int64
>>> s
0    1
1    2
2    3
3    4
4    5
dtype: int64


>>> import cudf
>>> cudf.set_option("copy_on_write", True)
>>> s = cudf.Series([1, 2, 3, 4, 5])
>>> s1 = s[0:2]
>>> s1[0] = 10
>>> s1
0    10
1     2
dtype: int64
>>> s
0    1
1    2
2    3
3    4
4    5
dtype: int64
```
2. There are numerous other inconsistencies, which are solved by copy-on-write. Read more about them [here](https://phofl.github.io/cow-introduction.html).


## How to disable it


Copy-on-write can be disable by setting ``copy_on_write`` cudf option to ``False``:

```python
>>> cudf.set_option("copy_on_write", False)
```
