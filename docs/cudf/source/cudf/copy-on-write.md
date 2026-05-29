(copy-on-write-user-doc)=

# Copy-on-write

Copy-on-write is a memory management strategy that allows multiple cuDF objects to share the the same underlying data if they were all derived from
operations that do not modify the data. For example, making a contiguous slice of a `DataFrame` or `Series` object will share the underlying data
with the original object.
However, when either the original or derived object is _modified_, a copy of the data is made prior to the modification, ensuring that the changes do not propagate between the two objects.

Copy-on-write is the default behavior since 26.08, aligning with [pandas 3.0](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).

## Making copies

Copy-on-write is an implicit behavior that requires no user API changes.

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

Therefore, you should no longer need to defensively `copy()` a cuDF object after deriving it from another cuDF object as any modifications
will no longer propagate.
