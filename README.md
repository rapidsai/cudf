# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cudf/job/branches/job/cudf-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cudf/job/branches/job/cudf-branch-pipeline/)

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/cudf/blob/main/README.md) ensure you are on the `main` branch.

Built based on the [Apache Arrow](http://arrow.apache.org/) columnar memory format, cuDF is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data.

cuDF provides a pandas-like API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows without going into the details of CUDA programming.

For example, the following snippet downloads a CSV, then uses the GPU to parse it into rows and columns and run calculations:
```python
import cudf, io, requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode('utf-8')

tips_df = cudf.read_csv(StringIO(content))
tips_df['tip_percentage'] = tips_df['tip'] / tips_df['total_bill'] * 100

# display average tip by dining party size
print(tips_df.groupby('size').tip_percentage.mean())
```

Output:
```
size
1    21.729201548727808
2    16.571919173482897
3    15.215685473711837
4    14.594900639351332
5    14.149548965142023
6    15.622920072028379
Name: tip_percentage, dtype: float64
```

For additional examples, browse our complete [API documentation](https://docs.rapids.ai/api/cudf/stable/), or check out our more detailed [notebooks](https://github.com/rapidsai/notebooks-contrib).

## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize cuDF.

## Installation


### CUDA/GPU requirements

* CUDA 10.0+
* NVIDIA driver 410.48+
* Pascal architecture or better (Compute Capability >=6.0)

### Conda

cuDF can be installed with conda ([miniconda](https://conda.io/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

For `cudf version == 0.13` :
```bash
# for CUDA 10.1
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=0.13 python=3.7 cudatoolkit=10.1

# or, for CUDA 10.2
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=0.13 python=3.7 cudatoolkit=10.2

```

For the nightly version of `cudf` :
```bash
# for CUDA 10.1
conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
    cudf python=3.7 cudatoolkit=10.1

# or, for CUDA 10.2
conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge \
    cudf python=3.7 cudatoolkit=10.2
```

Note: cuDF is supported only on Linux, and with Python versions 3.7 and later.

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
