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

10 Minutes to cuDF and Dask-cuDF
=======================

Modeled after 10 Minutes to Pandas, this is a short introduction to cuDF and Dask-cuDF, geared mainly for new users.

### What are these Libraries?

[cuDF](https://github.com/rapidsai/cudf) is a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating tabular data using a DataFrame style API.

[Dask](https://dask.org/) is a flexible library for parallel computing in Python that makes scaling out your workflow smooth and simple. On the CPU, Dask uses Pandas to execute operations in parallel on DataFrame partitions.

[Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) extends Dask where necessary to allow its DataFrame partitions to be processed by cuDF GPU DataFrames as opposed to Pandas DataFrames. For instance, when you call dask_cudf.read_csv(...), your cluster’s GPUs do the work of parsing the CSV file(s) with underlying cudf.read_csv().


### When to use cuDF and Dask-cuDF

If your workflow is fast enough on a single GPU or your data comfortably fits in memory on a single GPU, you would want to use cuDF. If you want to distribute your workflow across multiple GPUs, have more data than you can fit in memory on a single GPU, or want to analyze data spread across many files at once, you would want to use Dask-cuDF.

```{code-cell} ipython3
import os

import cupy as cp
import pandas as pd
import cudf
import dask_cudf

cp.random.seed(12)

#### Portions of this were borrowed and adapted from the
#### cuDF cheatsheet, existing cuDF documentation,
#### and 10 Minutes to Pandas.
```

Object Creation
---------------

+++

Creating a `cudf.Series` and `dask_cudf.Series`.

```{code-cell} ipython3
s = cudf.Series([1,2,3,None,4])
s
```

```{code-cell} ipython3
ds = dask_cudf.from_cudf(s, npartitions=2) 
ds.compute()
```

Creating a `cudf.DataFrame` and a `dask_cudf.DataFrame` by specifying values for each column.

```{code-cell} ipython3
df = cudf.DataFrame({'a': list(range(20)),
                     'b': list(reversed(range(20))),
                     'c': list(range(20))
                    })
df
```

```{code-cell} ipython3
ddf = dask_cudf.from_cudf(df, npartitions=2) 
ddf.compute()
```

Creating a `cudf.DataFrame` from a pandas `Dataframe` and a `dask_cudf.Dataframe` from a `cudf.Dataframe`.

*Note that best practice for using Dask-cuDF is to read data directly into a `dask_cudf.DataFrame` with something like `read_csv` (discussed below).*

```{code-cell} ipython3
pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
gdf = cudf.DataFrame.from_pandas(pdf)
gdf
```

```{code-cell} ipython3
dask_gdf = dask_cudf.from_cudf(gdf, npartitions=2)
dask_gdf.compute()
```

Viewing Data
-------------

+++

Viewing the top rows of a GPU dataframe.

```{code-cell} ipython3
df.head(2)
```

```{code-cell} ipython3
ddf.head(2)
```

Sorting by values.

```{code-cell} ipython3
df.sort_values(by='b')
```

```{code-cell} ipython3
ddf.sort_values(by='b').compute()
```

Selection
------------

## Getting

+++

Selecting a single column, which initially yields a `cudf.Series` or `dask_cudf.Series`. Calling `compute` results in a `cudf.Series` (equivalent to `df.a`).

```{code-cell} ipython3
df['a']
```

```{code-cell} ipython3
ddf['a'].compute()
```

## Selection by Label

+++

Selecting rows from index 2 to index 5 from columns 'a' and 'b'.

```{code-cell} ipython3
df.loc[2:5, ['a', 'b']]
```

```{code-cell} ipython3
ddf.loc[2:5, ['a', 'b']].compute()
```

## Selection by Position

+++

Selecting via integers and integer slices, like numpy/pandas. Note that this functionality is not available for Dask-cuDF DataFrames.

```{code-cell} ipython3
df.iloc[0]
```

```{code-cell} ipython3
df.iloc[0:3, 0:2]
```

You can also select elements of a `DataFrame` or `Series` with direct index access.

```{code-cell} ipython3
df[3:5]
```

```{code-cell} ipython3
s[3:5]
```

## Boolean Indexing

+++

Selecting rows in a `DataFrame` or `Series` by direct Boolean indexing.

```{code-cell} ipython3
df[df.b > 15]
```

```{code-cell} ipython3
ddf[ddf.b > 15].compute()
```

Selecting values from a `DataFrame` where a Boolean condition is met, via the `query` API.

```{code-cell} ipython3
df.query("b == 3")
```

```{code-cell} ipython3
ddf.query("b == 3").compute()
```

You can also pass local variables to Dask-cuDF queries, via the `local_dict` keyword. With standard cuDF, you may either use the `local_dict` keyword or directly pass the variable via the `@` keyword. Supported logical operators include `>`, `<`, `>=`, `<=`, `==`, and `!=`.

```{code-cell} ipython3
cudf_comparator = 3
df.query("b == @cudf_comparator")
```

```{code-cell} ipython3
dask_cudf_comparator = 3
ddf.query("b == @val", local_dict={'val':dask_cudf_comparator}).compute()
```

Using the `isin` method for filtering.

```{code-cell} ipython3
df[df.a.isin([0, 5])]
```

## MultiIndex

+++

cuDF supports hierarchical indexing of DataFrames using MultiIndex. Grouping hierarchically (see `Grouping` below) automatically produces a DataFrame with a MultiIndex.

```{code-cell} ipython3
arrays = [['a', 'a', 'b', 'b'], [1, 2, 3, 4]]
tuples = list(zip(*arrays))
idx = cudf.MultiIndex.from_tuples(tuples)
idx
```

This index can back either axis of a DataFrame.

```{code-cell} ipython3
gdf1 = cudf.DataFrame({'first': cp.random.rand(4), 'second': cp.random.rand(4)})
gdf1.index = idx
gdf1
```

```{code-cell} ipython3
gdf2 = cudf.DataFrame({'first': cp.random.rand(4), 'second': cp.random.rand(4)}).T
gdf2.columns = idx
gdf2
```

Accessing values of a DataFrame with a MultiIndex. Note that slicing is not yet supported.

```{code-cell} ipython3
gdf1.loc[('b', 3)]
```

Missing Data
------------

+++

Missing data can be replaced by using the `fillna` method.

```{code-cell} ipython3
s.fillna(999)
```

```{code-cell} ipython3
ds.fillna(999).compute()
```

Operations
------------

+++

## Stats

+++

Calculating descriptive statistics for a `Series`.

```{code-cell} ipython3
s.mean(), s.var()
```

```{code-cell} ipython3
ds.mean().compute(), ds.var().compute()
```

## Applymap

+++

Applying functions to a `Series`. Note that applying user defined functions directly with Dask-cuDF is not yet implemented. For now, you can use [map_partitions](http://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html) to apply a function to each partition of the distributed dataframe.

```{code-cell} ipython3
def add_ten(num):
    return num + 10

df['a'].applymap(add_ten)
```

```{code-cell} ipython3
ddf['a'].map_partitions(add_ten).compute()
```

## Histogramming

+++

Counting the number of occurrences of each unique value of variable.

```{code-cell} ipython3
df.a.value_counts()
```

```{code-cell} ipython3
ddf.a.value_counts().compute()
```

## String Methods

+++

Like pandas, cuDF provides string processing methods in the `str` attribute of `Series`. Full documentation of string methods is a work in progress. Please see the cuDF API documentation for more information.

```{code-cell} ipython3
s = cudf.Series(['A', 'B', 'C', 'Aaba', 'Baca', None, 'CABA', 'dog', 'cat'])
s.str.lower()
```

```{code-cell} ipython3
ds = dask_cudf.from_cudf(s, npartitions=2)
ds.str.lower().compute()
```

## Concat

+++

Concatenating `Series` and `DataFrames` row-wise.

```{code-cell} ipython3
s = cudf.Series([1, 2, 3, None, 5])
cudf.concat([s, s])
```

```{code-cell} ipython3
ds2 = dask_cudf.from_cudf(s, npartitions=2)
dask_cudf.concat([ds2, ds2]).compute()
```

## Join

+++

Performing SQL style merges. Note that the dataframe order is not maintained, but may be restored post-merge by sorting by the index.

```{code-cell} ipython3
df_a = cudf.DataFrame()
df_a['key'] = ['a', 'b', 'c', 'd', 'e']
df_a['vals_a'] = [float(i + 10) for i in range(5)]

df_b = cudf.DataFrame()
df_b['key'] = ['a', 'c', 'e']
df_b['vals_b'] = [float(i+100) for i in range(3)]

merged = df_a.merge(df_b, on=['key'], how='left')
merged
```

```{code-cell} ipython3
ddf_a = dask_cudf.from_cudf(df_a, npartitions=2)
ddf_b = dask_cudf.from_cudf(df_b, npartitions=2)

merged = ddf_a.merge(ddf_b, on=['key'], how='left').compute()
merged
```

## Append

+++

Appending values from another `Series` or array-like object.

```{code-cell} ipython3
s.append(s)
```

```{code-cell} ipython3
ds2.append(ds2).compute()
```

## Grouping

+++

Like pandas, cuDF and Dask-cuDF support the Split-Apply-Combine groupby paradigm.

```{code-cell} ipython3
df['agg_col1'] = [1 if x % 2 == 0 else 0 for x in range(len(df))]
df['agg_col2'] = [1 if x % 3 == 0 else 0 for x in range(len(df))]

ddf = dask_cudf.from_cudf(df, npartitions=2)
```

Grouping and then applying the `sum` function to the grouped data.

```{code-cell} ipython3
df.groupby('agg_col1').sum()
```

```{code-cell} ipython3
ddf.groupby('agg_col1').sum().compute()
```

Grouping hierarchically then applying the `sum` function to grouped data.

```{code-cell} ipython3
df.groupby(['agg_col1', 'agg_col2']).sum()
```

```{code-cell} ipython3
ddf.groupby(['agg_col1', 'agg_col2']).sum().compute()
```

Grouping and applying statistical functions to specific columns, using `agg`.

```{code-cell} ipython3
df.groupby('agg_col1').agg({'a':'max', 'b':'mean', 'c':'sum'})
```

```{code-cell} ipython3
ddf.groupby('agg_col1').agg({'a':'max', 'b':'mean', 'c':'sum'}).compute()
```

## Transpose

+++

Transposing a dataframe, using either the `transpose` method or `T` property. Currently, all columns must have the same type. Transposing is not currently implemented in Dask-cuDF.

```{code-cell} ipython3
sample = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
sample
```

```{code-cell} ipython3
sample.transpose()
```

Time Series
------------

+++

`DataFrames` supports `datetime` typed columns, which allow users to interact with and filter data based on specific timestamps.

```{code-cell} ipython3
import datetime as dt

date_df = cudf.DataFrame()
date_df['date'] = pd.date_range('11/20/2018', periods=72, freq='D')
date_df['value'] = cp.random.sample(len(date_df))

search_date = dt.datetime.strptime('2018-11-23', '%Y-%m-%d')
date_df.query('date <= @search_date')
```

```{code-cell} ipython3
date_ddf = dask_cudf.from_cudf(date_df, npartitions=2)
date_ddf.query('date <= @search_date', local_dict={'search_date':search_date}).compute()
```

Categoricals
------------

+++

`DataFrames` support categorical columns.

```{code-cell} ipython3
gdf = cudf.DataFrame({"id": [1, 2, 3, 4, 5, 6], "grade":['a', 'b', 'b', 'a', 'a', 'e']})
gdf['grade'] = gdf['grade'].astype('category')
gdf
```

```{code-cell} ipython3
dgdf = dask_cudf.from_cudf(gdf, npartitions=2)
dgdf.compute()
```

Accessing the categories of a column. Note that this is currently not supported in Dask-cuDF.

```{code-cell} ipython3
gdf.grade.cat.categories
```

Accessing the underlying code values of each categorical observation.

```{code-cell} ipython3
gdf.grade.cat.codes
```

```{code-cell} ipython3
dgdf.grade.cat.codes.compute()
```

Converting Data Representation
--------------------------------

+++

## Pandas

+++

Converting a cuDF and Dask-cuDF `DataFrame` to a pandas `DataFrame`.

```{code-cell} ipython3
df.head().to_pandas()
```

```{code-cell} ipython3
ddf.compute().head().to_pandas()
```

## Numpy

+++

Converting a cuDF or Dask-cuDF `DataFrame` to a numpy `ndarray`.

```{code-cell} ipython3
df.to_numpy()
```

```{code-cell} ipython3
ddf.compute().to_numpy()
```

Converting a cuDF or Dask-cuDF `Series` to a numpy `ndarray`.

```{code-cell} ipython3
df['a'].to_numpy()
```

```{code-cell} ipython3
ddf['a'].compute().to_numpy()
```

## Arrow

+++

Converting a cuDF or Dask-cuDF `DataFrame` to a PyArrow `Table`.

```{code-cell} ipython3
df.to_arrow()
```

```{code-cell} ipython3
ddf.compute().to_arrow()
```

Getting Data In/Out
------------------------

+++

## CSV

+++

Writing to a CSV file.

```{code-cell} ipython3
if not os.path.exists('example_output'):
    os.mkdir('example_output')
    
df.to_csv('example_output/foo.csv', index=False)
```

```{code-cell} ipython3
ddf.compute().to_csv('example_output/foo_dask.csv', index=False)
```

Reading from a csv file.

```{code-cell} ipython3
df = cudf.read_csv('example_output/foo.csv')
df
```

```{code-cell} ipython3
ddf = dask_cudf.read_csv('example_output/foo_dask.csv')
ddf.compute()
```

Reading all CSV files in a directory into a single `dask_cudf.DataFrame`, using the star wildcard.

```{code-cell} ipython3
ddf = dask_cudf.read_csv('example_output/*.csv')
ddf.compute()
```

## Parquet

+++

Writing to parquet files, using the CPU via PyArrow.

```{code-cell} ipython3
df.to_parquet('example_output/temp_parquet')
```

Reading parquet files with a GPU-accelerated parquet reader.

```{code-cell} ipython3
df = cudf.read_parquet('example_output/temp_parquet')
df
```

Writing to parquet files from a `dask_cudf.DataFrame` using PyArrow under the hood.

```{code-cell} ipython3
ddf.to_parquet('example_files')  
```

## ORC

+++

Reading ORC files.

```{code-cell} ipython3
import os
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath("__file__"))
cudf_root = Path(current_dir).parents[3]
file_path = os.path.join(cudf_root, "python", "cudf", "cudf", "tests", "data", "orc", "TestOrcFile.test1.orc")
file_path
```

```{code-cell} ipython3
df2 = cudf.read_orc(file_path)
df2
```

Dask Performance Tips
--------------------------------

Like Apache Spark, Dask operations are [lazy](https://en.wikipedia.org/wiki/Lazy_evaluation). Instead of being executed at that moment, most operations are added to a task graph and the actual evaluation is delayed until the result is needed.

Sometimes, though, we want to force the execution of operations. Calling `persist` on a Dask collection fully computes it (or actively computes it in the background), persisting the result into memory. When we're using distributed systems, we may want to wait until `persist` is finished before beginning any downstream operations. We can enforce this contract by using `wait`. Wrapping an operation with `wait` will ensure it doesn't begin executing until all necessary upstream operations have finished.

The snippets below provide basic examples, using `LocalCUDACluster` to create one dask-worker per GPU on the local machine. For more detailed information about `persist` and `wait`, please see the Dask documentation for [persist](https://docs.dask.org/en/latest/api.html#dask.persist) and [wait](https://docs.dask.org/en/latest/futures.html#distributed.wait). Wait relies on the concept of Futures, which is beyond the scope of this tutorial. For more information on Futures, see the Dask [Futures](https://docs.dask.org/en/latest/futures.html) documentation. For more information about multi-GPU clusters, please see the [dask-cuda](https://github.com/rapidsai/dask-cuda) library (documentation is in progress).

+++

First, we set up a GPU cluster. With our `client` set up, Dask-cuDF computation will be distributed across the GPUs in the cluster.

```{code-cell} ipython3
import time

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

cluster = LocalCUDACluster()
client = Client(cluster)
client
```

### Persisting Data
Next, we create our Dask-cuDF DataFrame and apply a transformation, storing the result as a new column.

```{code-cell} ipython3
nrows = 10000000

df2 = cudf.DataFrame({'a': cp.arange(nrows), 'b': cp.arange(nrows)})
ddf2 = dask_cudf.from_cudf(df2, npartitions=5)
ddf2['c'] = ddf2['a'] + 5
ddf2
```

```{code-cell} ipython3
!nvidia-smi
```

Because Dask is lazy, the computation has not yet occurred. We can see that there are twenty tasks in the task graph and we've used about 800 MB of memory. We can force computation by using `persist`. By forcing execution, the result is now explicitly in memory and our task graph only contains one task per partition (the baseline).

```{code-cell} ipython3
ddf2 = ddf2.persist()
ddf2
```

```{code-cell} ipython3
!nvidia-smi
```

Because we forced computation, we now have a larger object in distributed GPU memory.

+++

### Wait
Depending on our workflow or distributed computing setup, we may want to `wait` until all upstream tasks have finished before proceeding with a specific function. This section shows an example of this behavior, adapted from the Dask documentation.

First, we create a new Dask DataFrame and define a function that we'll map to every partition in the dataframe.

```{code-cell} ipython3
import random

nrows = 10000000

df1 = cudf.DataFrame({'a': cp.arange(nrows), 'b': cp.arange(nrows)})
ddf1 = dask_cudf.from_cudf(df1, npartitions=100)

def func(df):
    time.sleep(random.randint(1, 60))
    return (df + 5) * 3 - 11
```

This function will do a basic transformation of every column in the dataframe, but the time spent in the function will vary due to the `time.sleep` statement randomly adding 1-60 seconds of time. We'll run this on every partition of our dataframe using `map_partitions`, which adds the task to our task-graph, and store the result. We can then call `persist` to force execution.

```{code-cell} ipython3
results_ddf = ddf2.map_partitions(func)
results_ddf = results_ddf.persist()
```

However, some partitions will be done **much** sooner than others. If we had downstream processes that should wait for all partitions to be completed, we can enforce that behavior using `wait`.

```{code-cell} ipython3
wait(results_ddf)
```

## With `wait`, we can safely proceed on in our workflow.

```{code-cell} ipython3

```
