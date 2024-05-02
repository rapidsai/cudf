(copy-on-write-user-doc)=

# Copy-on-write

Copy-on-write is a memory management strategy that allows multiple cuDF objects containing the same data to refer to the same memory address as long as neither of them modify the underlying data.
With this approach, any operation that generates an unmodified view of an object (such as copies, slices, or methods like `DataFrame.head`) returns a new object that points to the same memory as the original.
However, when either the existing or new object is _modified_, a copy of the data is made prior to the modification, ensuring that the changes do not propagate between the two objects.
This behavior is best understood by looking at the examples below.

The default behaviour in cuDF is for copy-on-write to be disabled, so to use it, one must explicitly
opt in by setting a cuDF option. It is recommended to set the copy-on-write at the beginning of the
script execution, because when this setting is changed in middle of a script execution there will
be un-intended behavior where the objects created when copy-on-write is enabled will still have the
copy-on-write behavior whereas the objects created when copy-on-write is disabled will have
different behavior.

## Enabling copy-on-write

1. Use `cudf.set_option`:

    ```python
    >>> import cudf
    >>> cudf.set_option("copy_on_write", True)
    ```

2. Set the environment variable ``CUDF_COPY_ON_WRITE`` to ``1`` prior to the
launch of the Python interpreter:

    ```bash
    export CUDF_COPY_ON_WRITE="1" python -c "import cudf"
    ```

## Disabling copy-on-write


Copy-on-write can be disabled by setting the ``copy_on_write`` option to ``False``:

```python
>>> cudf.set_option("copy_on_write", False)
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

When copy-on-write is enabled, there is no longer a concept of views when
slicing or indexing. In this sense, indexing behaves as one would expect for
built-in Python containers like `lists`, rather than indexing `numpy arrays`.
Modifying a "view" created by cuDF will always trigger a copy and will not
modify the original object.

Copy-on-write produces much more consistent copy semantics. Since every object is a copy of the original, users no longer have to think about when modifications may unexpectedly happen in place. This will bring consistency across operations and bring cudf and pandas behavior into alignment when copy-on-write is enabled for both. Here is one example where pandas and cudf are currently inconsistent without copy-on-write enabled:

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

The above inconsistency is solved when copy-on-write is enabled:

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


### Explicit deep and shallow copies comparison


|                     | Copy-on-Write enabled                                                                                                                                                                                          | Copy-on-Write disabled (default)                                                                               |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `.copy(deep=True)`  | A true copy is made and changes don't propagate to the original object.                                                                                                                            | A true copy is made and changes don't propagate to the original object.                  |
| `.copy(deep=False)` | Memory is shared between the two objects and but any write operation on one object will trigger a true physical copy before the write is performed. Hence changes will not propagate to the original object. | Memory is shared between the two objects and changes performed on one will propagate to the other object. |
