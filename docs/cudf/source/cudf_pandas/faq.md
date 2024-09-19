# FAQ and Known Issues

## When should I use `cudf.pandas` vs using the cuDF library directly?

`cudf.pandas` is the quickest and easiest way to get pandas code
running on the GPU. However, there are some situations in which using
the cuDF library directly should be considered.

- cuDF implements a subset of the pandas API, while `cudf.pandas` will
  fall back automatically to pandas as needed. If you can write your
  code to use just the operations supported by cuDF, you will benefit
  from increased performance by using cuDF directly.

- cuDF does offer some functions and methods that pandas does not. For
  example, cuDF has a [`.list`
  accessor](https://docs.rapids.ai/api/cudf/stable/api_docs/list_handling/)
  for working with list-like data. If you need access to the
  additional functionality in cuDF, you will need to use the cuDF
  package directly.

## How closely does this match pandas?

You can use 100% of the pandas API and most things will work
identically to pandas.

`cudf.pandas` is tested against the entire pandas unit test suite.
Currently, we're passing **93%** of the 187,000+ unit tests, with the
goal of passing 100%. Test failures are typically for edge cases and
due to the small number of behavioral differences between cuDF and
pandas. You can learn more about these edge cases in
[Known Limitations](#are-there-any-known-limitations)

We also run nightly tests that track interactions between
`cudf.pandas` and other third party libraries. See
[Third-Party Library Compatibility](#does-cudf-pandas-work-with-third-party-libraries).

## How can I tell if `cudf.pandas` is active?

You shouldn't have to write any code differently depending on whether
`cudf.pandas` is in use or not. You should use `pandas` and things
should just work.

In a few circumstances during testing and development however, you may
want to explicitly verify that `cudf.pandas` is active. To do that,
print the pandas module in your code and review the output; it should
look something like this:

```python
%load_ext cudf.pandas
import pandas as pd

print(pd)
<module 'pandas' (ModuleAccelerator(fast=cudf, slow=pandas))>
```

## Which functions will run on the GPU?

Generally, `cudf.pandas` will accelerate all the features in the
{ref}`cuDF API <cudf-api>` on the GPU. There are some exceptions. For
example, some functions are GPU-accelerated by cuDF but do not support
every combination of keyword arguments. In cases like unsupported
keyword arguments, cuDF is not able to provide GPU acceleration and
`cudf.pandas` will fall back to the CPU.

The most accurate way to assess which functions run on the GPU is to try
running the code while using the `cudf.pandas` profiling features. The
profiler will indicate which functions ran on GPU / CPU. To improve
performance, try to use only functionality that can run entirely on GPU.
This helps reduce the number of memory transfers needed to fallback to
CPU.

## How can I improve performance of my workflow with `cudf.pandas`?

Most workflows will see significant performance improvements with
`cudf.pandas`. However, sometimes things can be slower than expected.
First, it's important to note that GPUs are good at parallel processing
of large amounts of data. Small data sizes may be slower on GPU than
CPU, because of the cost of data transfers. cuDF achieves the highest
performance with many rows of data. As a _very rough_ rule of thumb,
`cudf.pandas` shines on workflows with more than 10,000 - 100,000 rows
of data, depending on the algorithms, data types, and other factors.
Datasets that are several gigabytes in size and/or have millions of
rows are a great fit for `cudf.pandas`.

Here are some more tips to improve workflow performance:

- Reshape data so it is long rather than wide (more rows, fewer
  columns). This improves cuDF's ability to execute in parallel on the
  entire GPU!
- Avoid element-wise iteration and mutation. If you can, use pandas
  functions to manipulate an entire column at once rather than writing
  raw `for` loops that compute and assign.
- If your data is really an n-dimensional array with lots of columns
  where you aim to do lots of math (like adding matrices),
  [CuPy](https://cupy.dev/) or [NumPy](https://numpy.org/) may be a
  better choice than pandas or `cudf.pandas`. Array libraries are built
  for different use cases than DataFrame libraries, and will get optimal
  performance from using contiguous memory for multidimensional array
  storage. Use the `.values` method to convert a DataFrame or Series to
  an array.

(does-cudf-pandas-work-with-third-party-libraries)=
## Does `cudf.pandas` work with third-party libraries?

`cudf.pandas` is tested with numerous popular third-party libraries.
`cudf.pandas` will not only work but will accelerate pandas operations
within these libraries. As part of our CI/CD system, we currently test
common interactions with the following Python libraries:

| Library          | Status |
|------------------|--------|
| cuGraph          | ✅      |
| cuML             | ✅      |
| Hvplot           | ✅      |
| Holoview         | ✅      |
| Ibis             | ✅      |
| Joblib           | ❌      |
| NumPy            | ✅      |
| Matplotlib       | ✅      |
| Plotly           | ✅      |
| PyTorch          | ✅      |
| Seaborn          | ✅      |
| Scikit-Learn     | ✅      |
| SciPy            | ✅      |
| Tensorflow       | ✅      |
| XGBoost          | ✅      |

Please review the section on [Known Limitations](#are-there-any-known-limitations)
for details about what is expected not to work (and why).

## Can I use `cudf.pandas` with Dask or PySpark?

`cudf.pandas` is not designed for distributed or out-of-core computing
(OOC) workflows today. If you are looking for accelerated OOC and
distributed solutions for data processing we recommend Dask and Apache
Spark.

Both Dask and Apache Spark support accelerated computing through configuration
based interfaces. Dask allows you to [configure the dataframe
backend](https://docs.dask.org/en/latest/how-to/selecting-the-collection-backend.html) to use
cuDF (learn more in [this
blog](https://medium.com/rapids-ai/easy-cpu-gpu-arrays-and-dataframes-run-your-dask-code-where-youd-like-e349d92351d)) and the [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/)
provides a similar configuration-based plugin for Spark.

(are-there-any-known-limitations)=
## Are there any known limitations?

There are a few known limitations that you should be aware of:

- Because fallback involves copying data from GPU to CPU and back,
  [value mutability](https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html#mutability-and-copying-of-data)
  of Pandas objects is not always guaranteed. You should follow the
  pandas recommendation to favor immutable operations.
- For performance reasons, joins and join-based operations are not
  currently implemented to maintain the same row ordering as standard
  pandas
- `cudf.pandas` isn't compatible with directly using `import cudf`
   and is intended to be used with pandas-based workflows.
- Unpickling objects that were pickled with "regular" pandas will not
  work: you must have pickled an object with `cudf.pandas` enabled for
  it to be unpickled when `cudf.pandas` is enabled.
- Global variables can be accessed but can't be modified during CPU-fallback

  ```python
   %load_ext cudf.pandas
   import pandas as pd

   lst = [10]

   def udf(x):
       lst.append(x)
       return x + lst[0]

   s = pd.Series(range(2)).apply(udf)
   print(s) # we can access the value in lst
   0    10
   1    11
   dtype: int64
   print(lst) # lst is unchanged, as this specific UDF could not run on the GPU
   [10]
   ```
- `cudf.pandas` (and cuDF in general) is only compatible with pandas 2. Version
  24.02 of cudf was the last to support pandas 1.5.x.

## Can I force running on the CPU?

To run your code on CPU, just run without activating `cudf.pandas`,
and "regular pandas" will be used.

If needed, GPU acceleration may be disabled when using `cudf.pandas`
for testing or benchmarking purposes. To do so, set the
`CUDF_PANDAS_FALLBACK_MODE` environment variable, e.g.

```bash
CUDF_PANDAS_FALLBACK_MODE=1 python -m cudf.pandas some_script.py
```
