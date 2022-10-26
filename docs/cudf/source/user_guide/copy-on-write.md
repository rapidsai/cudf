# Copy on write

Copy on write enables ability to save on GPU memory usage when deep copies of a column
are made.

## How to enable it

i. Either by using `set_option` in `cudf`:

```python
>>> import cudf
>>> cudf.set_option("copy_on_write", True)
```

ii. Or, by setting an environment variable ``CUDF_COPY_ON_WRITE`` to ``1`` prior to the
launch of the python interpreter:

```bash
export CUDF_COPY_ON_WRITE="1"
```


## Making copies

There are no additional changes required in the code to make use of copy-on-write.

```python
>>> series = cudf.Series([1, 2, 3, 4])
```

Performing a deep copy will create a new series object but pointing to the
same underlying device memory:

```python
>>> copied_series = series.copy(deep=True)
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
>>> 
>>> 
>>> series.data.ptr
140102175031296
>>> copied_series.data.ptr
140102175031296
```

But, when there is a write-operation being performed on either ``series`` or
``copied_series``, a true deep-copy is triggered:

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

Notice the underlying data these both series objects now point to completely
different device objects:

```python
>>> series.data.ptr
140102175032832
>>> copied_series.data.ptr
140102175031296
```

````{Warning}
When ``copy_on_write`` is enabled, all of the deep copies are constructed with 
weak-references, and it is recommended to not hand-construct the contents of `__cuda_array_interface__`, instead please use the `series.__cuda_array_interface`
or `series.data.__cuda_array_interface__` which will then take care of detaching any existing weak-references that a column contains.
````


## How to disable it


Copy on write can be disable by setting ``copy_on_write`` cudf option to ``False``:

```python
>>> cudf.set_option("copy_on_write", False)
```