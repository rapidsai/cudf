GroupBy
=======

cuDF's supports a small (but important) subset of
Pandas' [groupby API](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html).

## Summary of supported operations

1. Grouping by one or more columns
1. Basic aggregations such as "sum", "mean", etc.
1. Quantile aggregation
1. A "collect" or `list` aggregation for collecting values in a group into lists
1. Automatic exclusion of "nuisance" columns when aggregating
1. Iterating over the groups of a GroupBy object
1. `GroupBy.groups` API that returns a mapping of group keys to row labels
1. `GroupBy.apply` API for performing arbitrary operations on each group. Note that
   this has very limited functionality compared to the equivalent Pandas function.
   See the section on [apply](#groupby-apply) for more details.
1. `GroupBy.pipe` similar to [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls).

## Grouping

A GroupBy object is created by grouping the values of a `Series` or `DataFrame`
by one or more columns:

```python
import cudf

>>> df = cudf.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
>>> df
>>> gb1 = df.groupby('a')  # grouping by a single column
>>> gb2 = df.groupby(['a', 'b'])  # grouping by multiple columns
>>> gb3 = df.groupby(cudf.Series(['a', 'a', 'b', 'b', 'b']))  # grouping by an external column
```

### Grouping by index levels

You can also group by one or more levels of a MultiIndex:

```python
>>> df = cudf.DataFrame(
...     {'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]}
... ).set_index(['a', 'b'])
...
>>> df.groupby(level='a')
```

### The `Grouper` object

A `Grouper` can be used to disambiguate between columns and levels when they have the same name:

```python
>>> df
   b  c
b
1  1  1
1  1  2
1  2  3
2  2  4
2  3  5
>>> df.groupby('b', level='b')  # ValueError: Cannot specify both by and level
>>> df.groupby([cudf.Grouper(key='b'), cudf.Grouper(level='b')])  # OK
```

## Aggregating

The following table summarizes the supported aggregations
and the dtypes on which they are supported.

| Aggregations\dtypes | Numeric  | Datetime | String   | Categorical | List |
| ------------------- | -------- | -------  | -------- | ----------- | ---- |
| count               | ✅       | ✅       | ✅       | ✅          |      |
| size                | ✅       | ✅       | ✅       | ✅          |      |
| sum                 | ✅       | ✅       |          |             |      |
| idxmin              | ✅       | ✅       |          |             |      |
| idxmax              | ✅       | ✅       |          |             |      |
| min                 | ✅       | ✅       | ✅       |             |      |
| max                 | ✅       | ✅       | ✅       |             |      |
| mean                | ✅       | ✅       |          |             |      |
| var                 | ✅       | ✅       |          |             |      |
| std                 | ✅       | ✅       |          |             |      |
| quantile            | ✅       | ✅       |          |             |      |
| median              | ✅       | ✅       |          |             |      |
| nunique             | ✅       | ✅       | ✅       | ✅          |      |
| nth                 | ✅       | ✅       | ✅       |             |      |
| collect             | ✅       | ✅       | ✅       |             | ✅   |

## GroupBy apply

## Rolling.groupby()
