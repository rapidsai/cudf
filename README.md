# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

### ❓ What is a GPU DataFrame?
A GPU DataFrame is like a pandas DataFrame (a table of data with rows and columns) but powered by your computer's Graphics Processing Unit (GPU) instead of the Central Processing Unit (CPU). 

**What's happening under the hood?**
- In pandas, operations (like sorting, filtering, or calculations) run on the CPU, which processes tasks one at a time or in small batches.
- In cuDF, these operations are parallelized across thousands of GPU cores, making them much faster for big datasets. Data is loaded into GPU memory, processed in parallel (e.g., calculating a new column for millions of rows at once), and results are returned.
- No major code changes needed—cuDF mimics pandas' API, so you can often just replace `import pandas as pd` with `import cudf as pd`.

This is built on NVIDIA's CUDA technology and libraries like libcudf for low-level GPU handling. If you're new to GPUs, think of it as upgrading from a single-lane road (CPU) to a massive highway (GPU) for data traffic.

### Why Use cuDF?
cuDF accelerates pandas-like data processing on NVIDIA GPUs, ideal for large datasets in data science, machine learning, or ETL pipelines. It uses the same API as pandas, so you can switch with minimal code changes.

**Speed Benefits**
-Large Data (100 Million Rows): cuDF is 19.17x faster than pandas, leveraging GPU parallelism for a significant speedup.
-Small Data (1000 Rows): pandas is much faster. cuDF's performance is hampered by the overhead of transferring the small dataset to the GPU.
-Conclusion: The performance benefit of cuDF is only realized on large datasets where the GPU's processing power outweighs the data transfer costs.

![Performance](https://i.ibb.co/0ybhzYMv/Screenshot-2025-10-01-123714.png)

Fig 1: On small data (1k rows), pandas is faster due to GPU overhead.

![Performance](https://i.ibb.co/nK5kvWH/Screenshot-2025-10-01-145759.png)

Fig 2: On large data (10M rows), cuDF is over 19x faster.

**Try It Yourself**
Run this code to compare speeds on your system:

```python
import time, pandas as pd, cudf
# Generate 100M rows (~1.5GB)
df = pd.DataFrame({'a': range(100_000_000), 'b': range(100_000_000)})
df.to_csv("large.csv", index=False)

# Pandas
start = time.time()
pdf = pd.read_csv("large.csv")
print(f"Pandas read: {time.time() - start:.4f} s")

# cuDF
start = time.time()
gdf = cudf.read_csv("large.csv")
print(f"cuDF read: {time.time() - start:.4f} s")
```

## 📢 cuDF can now be used as a no-code-change accelerator for pandas! To learn more, see [here](https://rapids.ai/cudf-pandas/)!

cuDF (pronounced "KOO-dee-eff") is a GPU DataFrame library
for loading, joining, aggregating, filtering, and otherwise
manipulating data. cuDF leverages
[libcudf](https://docs.rapids.ai/api/libcudf/stable/), a
blazing-fast C++/CUDA dataframe library and the [Apache
Arrow](https://arrow.apache.org/) columnar format to provide a
GPU-accelerated pandas API.

You can import `cudf` directly and use it like `pandas`:

```python
import cudf

tips_df = cudf.read_csv("https://github.com/plotly/datasets/raw/master/tips.csv")
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
print(tips_df.groupby("size").tip_percentage.mean())
```

Or, you can use cuDF as a no-code-change accelerator for pandas, using
[`cudf.pandas`](https://docs.rapids.ai/api/cudf/stable/cudf_pandas).
`cudf.pandas` supports 100% of the pandas API, utilizing cuDF for
supported operations and falling back to pandas when needed:

```python
%load_ext cudf.pandas  # pandas operations now use the GPU!

import pandas as pd

tips_df = pd.read_csv("https://github.com/plotly/datasets/raw/master/tips.csv")
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
print(tips_df.groupby("size").tip_percentage.mean())
```

## Resources

- [Try cudf.pandas now](https://nvda.ws/rapids-cudf): Explore `cudf.pandas` on a free GPU enabled instance on Google Colab!
- [Install](https://docs.rapids.ai/install): Instructions for installing cuDF and other [RAPIDS](https://rapids.ai) libraries.
- [cudf (Python) documentation](https://docs.rapids.ai/api/cudf/stable/)
- [libcudf (C++/CUDA) documentation](https://docs.rapids.ai/api/libcudf/stable/)
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help, contribute, and collaborate.

See the [RAPIDS install page](https://docs.rapids.ai/install) for
the most up-to-date information and commands for installing cuDF
and other RAPIDS packages.

## Installation

### CUDA/GPU requirements

* CUDA 12.0+ with a compatible NVIDIA driver
* Volta architecture or better (Compute Capability >=7.0)

### Pip

cuDF can be installed via `pip` from the NVIDIA Python Package Index.
Be sure to select the appropriate cuDF package depending
on the major version of CUDA available in your environment:

```bash
# CUDA 13
pip install cudf-cu13

# CUDA 12
pip install cudf-cu12
```

### Conda

cuDF can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)) from the `rapidsai` channel:

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge cudf=25.12 cuda-version=13.0

# CUDA 12
conda install -c rapidsai -c conda-forge cudf=25.12 cuda-version=12.9
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: cuDF is supported only on Linux, and with Python versions 3.10 and later.

See the [RAPIDS installation guide](https://docs.rapids.ai/install) for more OS and version info.

## Build/Install from Source
See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing

Please see our [guide for contributing to cuDF](CONTRIBUTING.md).
