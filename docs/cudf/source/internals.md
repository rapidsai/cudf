cuDF internals
==============

ColumnAccessor
--------------

cuDF  `Series`, `DataFrame` and `Index` are all subclasses of an internal `Frame` class.
The underlying data structure of `Frame` is a `ColumnAccessor`,
which can be accessed via the `._data` attribute:

```python
>>> a = cudf.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
>>> a._data
ColumnAccessor(OrderedColumnDict([('x', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d12e050>), ('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d12e320>)]), multiindex=False, level_names=(None,))
```

`ColumnAccessor` is an ordered, dict-like object that maps column labels to columns.
In addition, it supports things like selecting multiple columns (both by index and label),
as well as hierarchical indexing.

```python
>>> from cudf.core.column_accessor import ColumnAccessor
```

A ColumnAccessor behaves like an OrderedDict,
its values are coerced to Columns during construction:

```python
>>> ca = ColumnAccessor({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
>>> ca['x']
<cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>
>>> ca['y']
<cudf.core.column.string.StringColumn object at 0x7f5a7d578b90>
>>> ca.pop('x')
<cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>
>>> ca
ColumnAccessor(OrderedColumnDict([('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578b90>)]), multiindex=False, level_names=(None,))
```

Columns can be inserted at a specified location:

```python
>>> ca.insert('z', [3, 4, 5], loc=1)
>>> ca
ColumnAccessor(OrderedColumnDict([('x', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d578dd0>), ('z', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d578680>), ('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d12e3b0>)]), multiindex=False, level_names=(None,))
```

Selecting columns by index:

```python
>>> ca = ColumnAccessor({'x': [1, 2, 3], 'y': ['a', 'b', 'c'], 'z': [4, 5, 6]})
>>> ca.select_by_index(1)
ColumnAccessor(OrderedColumnDict([('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578830>)]), multiindex=False, level_names=(None,))
>>> ca.select_by_index([0, 1])
ColumnAccessor(OrderedColumnDict([('x', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>), ('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578830>)]), multiindex=False, level_names=(None,))    
>>> ca.select_by_index(slice(1, 3))
ColumnAccessor(OrderedColumnDict([('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578830>), ('z', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5788c0>)]), multiindex=False, level_names=(None,))
```

Selecting columns by label:

```python
>>> ca.select_by_label(['y', 'z'])
ColumnAccessor(OrderedColumnDict([('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578830>), ('z', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5788c0>)]), multiindex=False, level_names=(None,))
>>> ca.select_by_label(slice('x', 'y'))
ColumnAccessor(OrderedColumnDict([('x', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>), ('y', <cudf.core.column.string.StringColumn object at 0x7f5a7d578830>)]), multiindex=False, level_names=(None,))
```

A ColumnAccessor with tuple keys (and constructed with `multiindex=True`)
can be hierarchically indexed:

```python
>>> ca = ColumnAccessor({('a', 'b'): [1, 2, 3], ('a', 'c'): [2, 3, 4], 'b': [4, 5, 6]}, multiindex=True)
>>> ca.select_by_label('a')
ColumnAccessor(OrderedColumnDict([('b', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>), ('c', <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d578dd0>)]), multiindex=False, level_names=(None,))
>>> ca.select_by_label(('a', 'b'))
ColumnAccessor(OrderedColumnDict([(('a', 'b'), <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d5789e0>)]), multiindex=False, level_names=(None,))
```

"Wildcard" indexing is also allowed:

```python
>>> ca = ColumnAccessor({('a', 'b'): [1, 2, 3], ('a', 'c'): [2, 3, 4], ('d', 'b'): [4, 5, 6]}, multiindex=True)
>>> ca.select_by_label((slice(None), 'b'))
ColumnAccessor(OrderedColumnDict([(('a', 'b'), <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d578830>), (('d', 'b'), <cudf.core.column.numerical.NumericalColumn object at 0x7f5a7d578680>)]), multiindex=True, level_names=(None, None))
```

Finally, ColumnAccessors can convert to Pandas `Index` or `MultiIndex` objects:

```python
>>> ca.to_pandas_index()
MultiIndex([('a', 'b'),
            ('a', 'c'),
            ('d', 'b')],
           )
```
