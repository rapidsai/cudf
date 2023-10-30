# Frequently Asked Questions

## How closely does this match pandas?

Every change to cuDF pandas accelerator mode is tested against the entire
pandas unit test suite. Currently, we’re passing **93%** of the 187,000+ unit
tests, with a goal of passing 100%.

For most pandas workflows, things will “just work”. In a small set of
scenarios, we may hit edge cases that throw errors or don’t perfectly match
standard pandas. You can learn more about these edge cases in
[Known Limitations](#known-limitations)


## How can we tell if cudf.pandas is active?

`cudf.pandas` will be active if you’ve loaded the extension or executed a Python
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
based interfaces. Dask allows you to configure the dataframe backend to use
cuDF (learn more in this blog) and the RAPIDS Accelerator for Apache Spark
provides a similar

## Are there any known limitations?

There are a few known limitations that users should be aware of, depending on
their workflow:


- `cudf.pandas` can’t smoothly interact with tools or interfaces that convert data
formats using the NumPy C API
  - For example, you can `torch.tensor(df.values)` but not `torch.from_numpy(df.values), as the latter uses the NumPy C API
- Joins are not currently guaranteed to maintain the same row ordering as
standard pandas
- `cudf.pandas` can’t currently directly convert a dataframe into PyArrow table cudf.pandas isn’t compatible with directly using `import cudf` in workflows and is intended for pandas users
- Global variables can - be accessed but can’t be modified during CPU-fallback

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

