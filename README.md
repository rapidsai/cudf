# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

## 📢 cuDF can now be used as a no-code-change accelerator for pandas! To learn more, see [here](https://rapids.ai/cudf-pandas/)!

cuDF is a GPU DataFrame library for loading joining, aggregating,
filtering, and otherwise manipulating data. cuDF leverages
[libcudf](https://docs.rapids.ai/api/libcudf/stable/), a
blazing-fast C++/CUDA dataframe library and the [Apache
Arrow](https://arrow.apache.org/) columnar format to provide a
GPU-accelerated pandas API.

You can import `cudf` directly and use it like `pandas`:

```python
import cudf
import requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode("utf-8")

tips_df = cudf.read_csv(StringIO(content))
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
import requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode("utf-8")

tips_df = pd.read_csv(StringIO(content))
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
print(tips_df.groupby("size").tip_percentage.mean())
```

## Resources

- [Try cudf.pandas now](https://nvda.ws/rapids-cudf): Explore `cudf.pandas` on a free GPU enabled instance on Google Colab!
- [Install](https://rapids.ai/start.html): Instructions for installing cuDF and other [RAPIDS](https://rapids.ai) libraries.
- [cudf (Python) documentation](https://docs.rapids.ai/api/cudf/stable/)
- [libcudf (C++/CUDA) documentation](https://docs.rapids.ai/api/libcudf/stable/)
- [RAPIDS Community](https://rapids.ai/community.html): Get help, contribute, and collaborate.

## Installation

### CUDA/GPU requirements

* CUDA 11.2+
* NVIDIA driver 450.80.02+
* Pascal architecture or better (Compute Capability >=6.0)

### Conda

cuDF can be installed with conda (via [miniconda](https://conda.io/miniconda.html) or the full [Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.02 python=3.10 cuda-version=11.8
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: cuDF is supported only on Linux, and with Python versions 3.9 and later.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info.

## Build/Install from Source
See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing

Please see our [guide for contributing to cuDF](CONTRIBUTING.md).

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
