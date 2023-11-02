# FAQ and Known Issues

## How closely does this match pandas?

Every change to cuDF pandas Accelerator Mode is tested against the entire
pandas unit test suite.  Currently, we're passing **93%** of the 187,000+ unit
tests, with a goal of passing 100%.

For most pandas workflows, things will "just work". In a small set of
scenarios, we may hit edge cases that throw errors or don't perfectly match
standard pandas. You can learn more about these edge cases in
[Known Limitations](#are-there-any-known-limitations)


## How can we tell if cudf.pandas is active?

`cudf.pandas` will be active if you've loaded the extension or executed a Python
script with the module option. You should keep using pandas and things will
just work.

In a few circumstances during workflow testing or library development, you may
want to explicitly verify that cudf.pandas is active. To do that, print the
pandas module in your code and review the output:

```python
%load_ext cudf.pandas
import pandas as pd

print(pd)
<module 'pandas' (ModuleAccelerator(fast=cudf, slow=pandas))>
```

## Does this work with third-party libraries?

While we do not guarantee `cudf.pandas` will work with every external library, we
do actively test compatibility with many popular third-party libraries that
operate on pandas objects. cudf.pandas will not only work but will
accelerate pandas operations within these libraries.

As part of our CI/CD system, we currently test compatibility with the
following Python libraries:


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


## Can I use this with Dask and PySpark?

cudf.pandas is not designed for distributed or out-of-core computing (OOC)
workflows today. If you are looking for accelerated OOC and distributed
solutions for data processing we recommend Dask and Apache Spark.

Both Dask and Apache Spark support accelerated computing through configuration
based interfaces. Dask allows you to [configure the dataframe
backend](https://docs.dask.org/en/latest/how-to/selecting-the-collection-backend.html) to use
cuDF (learn more in [this
blog](https://medium.com/rapids-ai/easy-cpu-gpu-arrays-and-dataframes-run-your-dask-code-where-youd-like-e349d92351d)) and the [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/)
provides a similar configuration-based plugin for Spark.


## Are there any known limitations?

There are a few known limitations that users should be aware of, depending on
their workflow:


- `cudf.pandas` can't smoothly interact with tools or interfaces that convert data
formats using the NumPy C API
  - For example, you can `torch.tensor(df.values)` but not `torch.from_numpy(df.values), as the latter uses the NumPy C API
- Joins are not currently guaranteed to maintain the same row ordering as standard pandas
- cudf.pandas isn't compatible with directly using `import cudf` in workflows and is intended for pandas-based workflows.
- Global variables can - be accessed but can't be modified during CPU-fallback

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
- `cudf.pandas` (and cuDF in general) is currently only compatible with pandas 1.5.x



## Can I force running on the CPU?

If needed, GPU acceleration may be disabled when using cudf.pandas for testing
or benchmarking purposes. To do so, set the `CUDF_PANDAS_FALLBACK_MODE`
environment variable, e.g.

```bash
CUDF_PANDAS_FALLBACK_MODE=1 python -m cudf.pandas some_script.py
```

## When should I use cudf.pandas vs using cudf directly?

- Although it largely mimics the pandas API, cuDF does offer some functionality
that is not directly supported by pandas. If you need access to this
functionality, you will need to use the `cudf` package directly because you
cannot use `cudf` and the `cudf.pandas` module in the same script

- If you know that all the functionality you require is supported by cudf and you
do not need the additional pandas compatibility promised by cuDF's pandas
compatibility mode (e.g. join ordering) then you will benefit from
increased performance by using `cudf` directly.
