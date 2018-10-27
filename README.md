# cuDF

### :warning: Repo is frozen until 10/26 for refactoring to cuDF, no new issues or PRs :warning:

[![Build Status](http://18.191.94.64/buildStatus/icon?job=pygdf-master)](http://18.191.94.64/job/pygdf-master/)&nbsp;&nbsp;[![Documentation Status](https://readthedocs.org/projects/pygdf/badge/?version=latest)](http://pygdf.readthedocs.io/en/latest/?badge=latest)

The RAPIDS cuDF library is a DataFrame manipulation library based on Apache Arrow that accelerates loading, filtering, and manipulation of data for model training data preparation. The Python bindings of the core-accelerated CUDA DataFrame manipulation primitives mirror the pandas interface for seamless onboarding of pandas users.


## Setup

### Conda

You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

You can install and update cuDF using the conda command:

```bash
conda install -c numba -c conda-forge -c rapidsai/label/dev -c defaults cudf=0.2.0a1
```

You can create and activate a development environment using the conda command:

```bash
conda env create --name cudf_dev --file conda_environments/testing_py35.yml
source activate cudf_dev
```

### Install from Source

To install cuDF from source, clone the repository and run the following commands:

```bash
git clone --recurse-submodules https://github.com/rapidsai/cudf.git
mkdir -p cudf/libgdf/build
cd cudf/libgdf/build
cmake .. -DHASH_JOIN=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j install
make copy_python
python setup.py install

cd ../../cudf
python setup.py install
```

A Dockerfile is provided for building and installing the cuDF master branch.

**Notes**:
* We test with and recommended installing [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
* Host's installed nvidia driver must support >= the specified CUDA version (9.2 by default).
* Alternative CUDA_VERSION should be specified via Docker [build-arg](https://docs.docker.com/engine/reference/commandline/build/#set-build-time-variables---build-arg)
* Alternate branches for cudf may be specified as Docker build-args CUDF_REPO. See Dockerfile for example.
* Ubuntu 16.04 is the default OS for this container. Alternate OSes may be specified as Docker build-arg LINUX_VERSION. See list of [available images](https://hub.docker.com/r/nvidia/cuda/).
* Python 3.6 is default, but other versions may be specified via PYTHON_VERSION build-arg
* GCC & G++ 5.x are default compiler versions, but other versions (which are supplied by the OS package manager) may be specified via CC and CXX build-args respectively
* numba (0.40.0), numpy (1.14.3), and pandas (0.20.3) versions are also configurable as build-args

From cudf project root, to build with defaults:
```
docker build -t cudf .
...
 ---> ec65aaa3d4b1
 Successfully built ec65aaa3d4b1
 Successfully tagged cudf:latest

docker run --runtime=nvidia -it cudf bash
/# source activate gdf
(gdf) root@3f689ba9c842:/# python -c "import cudf"
(gdf) root@3f689ba9c842:/#
```

### Pip

Currently, we don't support pip install yet.  Please use conda for the time being.

### Testing

This project uses [py.test](https://docs.pytest.org/en/latest/).

In the source root directory and with the development environment activated, run:

```bash
py.test
```

## Getting Started

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/),  choosing a tag based on the NVIDIA driver version you’re running, for example notebooks on how you can utilize cuDF.

## RAPIDS Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<div align="center"><img src="img/rapids_logo.png" width="40%"/></div>

### GPU Arrow

The GPU version of Arrow is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the [Apache Arrow](https://arrow.apache.org/) columnar data format on the GPU. Currently, a subset of the features in Arrow are supported.

<div align="center"><img src="img/rapids_arrow.png" width="80%"/></div>
