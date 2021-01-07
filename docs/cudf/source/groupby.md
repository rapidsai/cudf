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

### The grouper object

## Aggregating

## GroupBy apply

## Rolling.groupby()
