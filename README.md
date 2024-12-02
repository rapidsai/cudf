# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

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

* CUDA 11.2+
* NVIDIA driver 450.80.02+
* Volta architecture or better (Compute Capability >=7.0)

### Pip

cuDF can be installed via `pip` from the NVIDIA Python Package Index.
Be sure to select the appropriate cuDF package depending
on the major version of CUDA available in your environment:

For CUDA 11.x:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
```

For CUDA 12.x:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

### Conda

cuDF can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)) from the `rapidsai` channel:

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=25.02 python=3.12 cuda-version=12.5
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: cuDF is supported only on Linux, and with Python versions 3.10 and later.

See the [RAPIDS installation guide](https://docs.rapids.ai/install) for more OS and version info.

## Build/Install from Source
See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing

Please see our [guide for contributing to cuDF](CONTRIBUTING.md).
