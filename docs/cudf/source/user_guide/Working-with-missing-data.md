---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Working with missing data

+++

In this section, we will discuss missing (also referred to as `NA`) values in cudf. cudf supports having missing values in all dtypes. These missing values are represented by `<NA>`. These values are also referenced as "null values".

+++

1. [How to Detect missing values](#How-to-Detect-missing-values)
2. [Float dtypes and missing data](#Float-dtypes-and-missing-data)
3. [Datetimes](#Datetimes)
4. [Calculations with missing data](#Calculations-with-missing-data)
5. [Sum/product of Null/nans](#Sum/product-of-Null/nans)
6. [NA values in GroupBy](#NA-values-in-GroupBy)
7. [Inserting missing data](#Inserting-missing-data)
8. [Filling missing values: fillna](#Filling-missing-values:-fillna)
9. [Filling with cudf Object](#Filling-with-cudf-Object)
10. [Dropping axis labels with missing data: dropna](#Dropping-axis-labels-with-missing-data:-dropna)
11. [Replacing generic values](#Replacing-generic-values)
12. [String/regular expression replacement](#String/regular-expression-replacement)
13. [Numeric replacement](#Numeric-replacement)

+++

## How to Detect missing values

+++

To detect missing values, you can use `isna()` and `notna()` functions.

```{code-cell} ipython3
import cudf
import numpy as np
```

```{code-cell} ipython3
df = cudf.DataFrame({'a': [1, 2, None, 4], 'b':[0.1, None, 2.3, 17.17]})
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
df.isna()
```

```{code-cell} ipython3
df['a'].notna()
```

One has to be mindful that in Python (and NumPy), the nan's don’t compare equal, but None's do. Note that cudf/NumPy uses the fact that `np.nan != np.nan`, and treats `None` like `np.nan`.

```{code-cell} ipython3
None == None
```

```{code-cell} ipython3
np.nan == np.nan
```

So as compared to above, a scalar equality comparison versus a None/np.nan doesn’t provide useful information.


```{code-cell} ipython3
df['b'] == np.nan
```

```{code-cell} ipython3
s = cudf.Series([None, 1, 2])
```

```{code-cell} ipython3
s
```

```{code-cell} ipython3
s == None
```

```{code-cell} ipython3
s = cudf.Series([1, 2, np.nan], nan_as_null=False)
```

```{code-cell} ipython3
s
```

```{code-cell} ipython3
s == np.nan
```

## Float dtypes and missing data

+++

Because ``NaN`` is a float, a column of integers with even one missing values is cast to floating-point dtype. However this doesn't happen by default.

By default if a ``NaN`` value is passed to `Series` constructor, it is treated as `<NA>` value. 

```{code-cell} ipython3
cudf.Series([1, 2, np.nan])
```

Hence to consider a ``NaN`` as ``NaN`` you will have to pass `nan_as_null=False` parameter into `Series` constructor.

```{code-cell} ipython3
cudf.Series([1, 2, np.nan], nan_as_null=False)
```

## Datetimes

+++

For `datetime64` types, cudf doesn't support having `NaT` values. Instead these values which are specific to numpy and pandas are considered as null values(`<NA>`) in cudf. The actual underlying value of `NaT` is `min(int64)` and cudf retains the underlying value when converting a cudf object to pandas object.


```{code-cell} ipython3
import pandas as pd
datetime_series = cudf.Series([pd.Timestamp("20120101"), pd.NaT, pd.Timestamp("20120101")])
datetime_series
```

```{code-cell} ipython3
datetime_series.to_pandas()
```

any operations on rows having `<NA>` values in `datetime` column will result in `<NA>` value at the same location in resulting column:

```{code-cell} ipython3
datetime_series - datetime_series
```

## Calculations with missing data

+++

Null values propagate naturally through arithmetic operations between pandas objects.

```{code-cell} ipython3
df1 = cudf.DataFrame({'a':[1, None, 2, 3, None], 'b':cudf.Series([np.nan, 2, 3.2, 0.1, 1], nan_as_null=False)})
```

```{code-cell} ipython3
df2 = cudf.DataFrame({'a':[1, 11, 2, 34, 10], 'b':cudf.Series([0.23, 22, 3.2, None, 1])})
```

```{code-cell} ipython3
df1
```

```{code-cell} ipython3
df2
```

```{code-cell} ipython3
df1 + df2
```

While summing the data along a series, `NA` values will be treated as `0`.

```{code-cell} ipython3
df1['a']
```

```{code-cell} ipython3
df1['a'].sum()
```

Since `NA` values are treated as `0`, the mean would result to 2 in this case `(1 + 0 + 2 + 3 + 0)/5 = 2`

```{code-cell} ipython3
df1['a'].mean()
```

To preserve `NA` values in the above calculations, `sum` & `mean` support `skipna` parameter.
By default it's value is
set to `True`, we can change it to `False` to preserve `NA` values.

```{code-cell} ipython3
df1['a'].sum(skipna=False)
```

```{code-cell} ipython3
df1['a'].mean(skipna=False)
```

Cumulative methods like `cumsum` and `cumprod` ignore `NA` values by default.

```{code-cell} ipython3
df1['a'].cumsum()
```

To preserve `NA` values in cumulative methods, provide `skipna=False`.

```{code-cell} ipython3
df1['a'].cumsum(skipna=False)
```

## Sum/product of Null/nans

+++

The sum of an empty or all-NA Series of a DataFrame is 0.

```{code-cell} ipython3
cudf.Series([np.nan], nan_as_null=False).sum()
```

```{code-cell} ipython3
cudf.Series([np.nan], nan_as_null=False).sum(skipna=False)
```

```{code-cell} ipython3
cudf.Series([], dtype='float64').sum()
```

The product of an empty or all-NA Series of a DataFrame is 1.

```{code-cell} ipython3
cudf.Series([np.nan], nan_as_null=False).prod()
```

```{code-cell} ipython3
cudf.Series([np.nan], nan_as_null=False).prod(skipna=False)
```

```{code-cell} ipython3
cudf.Series([], dtype='float64').prod()
```

## NA values in GroupBy

+++

`NA` groups in GroupBy are automatically excluded. For example:

```{code-cell} ipython3
df1
```

```{code-cell} ipython3
df1.groupby('a').mean()
```

It is also possible to include `NA` in groups by passing `dropna=False`

```{code-cell} ipython3
df1.groupby('a', dropna=False).mean()
```

## Inserting missing data

+++

All dtypes support insertion of missing value by assignment. Any specific location in series can made null by assigning it to `None`.

```{code-cell} ipython3
series = cudf.Series([1, 2, 3, 4])
```

```{code-cell} ipython3
series
```

```{code-cell} ipython3
series[2] = None
```

```{code-cell} ipython3
series
```

## Filling missing values: fillna

+++

`fillna()` can fill in `NA` & `NaN` values with non-NA data.

```{code-cell} ipython3
df1
```

```{code-cell} ipython3
df1['b'].fillna(10)
```

## Filling with cudf Object

+++

You can also fillna using a dict or Series that is alignable. The labels of the dict or index of the Series must match the columns of the frame you wish to fill. The use case of this is to fill a DataFrame with the mean of that column.

```{code-cell} ipython3
import cupy as cp
dff = cudf.DataFrame(cp.random.randn(10, 3), columns=list('ABC'))
```

```{code-cell} ipython3
dff.iloc[3:5, 0] = np.nan
```

```{code-cell} ipython3
dff.iloc[4:6, 1] = np.nan
```

```{code-cell} ipython3
dff.iloc[5:8, 2] = np.nan
```

```{code-cell} ipython3
dff
```

```{code-cell} ipython3
dff.fillna(dff.mean())
```

```{code-cell} ipython3
dff.fillna(dff.mean()[1:3])
```

## Dropping axis labels with missing data: dropna

+++

Missing data can be excluded using `dropna()`:


```{code-cell} ipython3
df1
```

```{code-cell} ipython3
df1.dropna(axis=0)
```

```{code-cell} ipython3
df1.dropna(axis=1)
```

An equivalent `dropna()` is available for Series. 

```{code-cell} ipython3
df1['a'].dropna()
```

## Replacing generic values

+++

Often times we want to replace arbitrary values with other values.

`replace()` in Series and `replace()` in DataFrame provides an efficient yet flexible way to perform such replacements.

```{code-cell} ipython3
series = cudf.Series([0.0, 1.0, 2.0, 3.0, 4.0])
```

```{code-cell} ipython3
series
```

```{code-cell} ipython3
series.replace(0, 5)
```

We can also replace any value with a `<NA>` value.

```{code-cell} ipython3
series.replace(0, None)
```

You can replace a list of values by a list of other values:

```{code-cell} ipython3
series.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
```

You can also specify a mapping dict:

```{code-cell} ipython3
series.replace({0: 10, 1: 100})
```

For a DataFrame, you can specify individual values by column:

```{code-cell} ipython3
df = cudf.DataFrame({"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]})
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
df.replace({"a": 0, "b": 5}, 100)
```

## String/regular expression replacement

+++

cudf supports replacing string values using `replace` API:

```{code-cell} ipython3
d = {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", None, "d"]}
```

```{code-cell} ipython3
df = cudf.DataFrame(d)
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
df.replace(".", "A Dot")
```

```{code-cell} ipython3
df.replace([".", "b"], ["A Dot", None])
```

Replace a few different values (list -> list):

```{code-cell} ipython3
df.replace(["a", "."], ["b", "--"])
```

Only search in column 'b' (dict -> dict):

```{code-cell} ipython3
df.replace({"b": "."}, {"b": "replacement value"})
```

## Numeric replacement

+++

`replace()` can also be used similar to `fillna()`.

```{code-cell} ipython3
df = cudf.DataFrame(cp.random.randn(10, 2))
```

```{code-cell} ipython3
df[np.random.rand(df.shape[0]) > 0.5] = 1.5
```

```{code-cell} ipython3
df.replace(1.5, None)
```

Replacing more than one value is possible by passing a list.


```{code-cell} ipython3
df00 = df.iloc[0, 0]
```

```{code-cell} ipython3
df.replace([1.5, df00], [5, 10])
```

You can also operate on the DataFrame in place:


```{code-cell} ipython3
df.replace(1.5, None, inplace=True)
```

```{code-cell} ipython3
df
```
