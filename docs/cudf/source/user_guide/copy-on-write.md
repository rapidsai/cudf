# Copy on write

Copy on write enables ability to save on GPU memory usage when copies(`.copy(deep=False)`) of a column
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

Performing a shallow copy will create a new series object but pointing to the
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
>>> series.data.ptr
140102175031296
>>> copied_series.data.ptr
140102175031296
```

But, when there is a write-operation being performed on either ``series`` or
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

Notice the underlying data these both series objects now point to completely
different device objects:

```python
>>> series.data.ptr
140102175032832
>>> copied_series.data.ptr
140102175031296
```

````{Warning}
When ``copy_on_write`` is enabled, all of the shallow copies are constructed with
weak-references, and it is recommended to not hand-construct the contents of `__cuda_array_interface__`, instead please use the `series.__cuda_array_interface__`
or `series.data.__cuda_array_interface__` which will then take care of unlinking any existing weak-references that a column contains.
````

## Notes

When copy-on-write is enabled, there is no concept of views. i.e., modifying any view created inside cudf will not actually not modify
the original object it was viewing and thus a separate copy is created and then modified.

## Advantages

With copy-on-write enabled and by requesting `.copy(deep=False)`, the GPU memory usage can be reduced drastically if you are not performing
write operations on all of those copies. This will also increase the speed at which objects are created for execution of your ETL workflow.

## How to disable it


Copy on write can be disable by setting ``copy_on_write`` cudf option to ``False``:

```python
>>> cudf.set_option("copy_on_write", False)
```
