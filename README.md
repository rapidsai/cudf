# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

[![Build Status](http://18.191.94.64/buildStatus/icon?job=cudf-master)](http://18.191.94.64/job/cudf-master/)&nbsp;&nbsp;[![Documentation Status](https://readthedocs.org/projects/pygdf/badge/?version=latest)](http://pygdf.readthedocs.io/en/latest/?badge=latest)

The RAPIDS cuDF library is a GPU DataFrame manipulation library based on Apache Arrow that accelerates loading, filtering, and manipulation of data for model training data preparation. The Python bindings of the core-accelerated CUDA DataFrame manipulation primitives mirror the pandas interface for seamless onboarding of pandas users.

## Quick Start - Demo Container

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running, for example notebooks on how you can utilize cuDF.

## Install cuDF

### Conda

You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

You can install and update cuDF using the conda command:

```bash
conda install -c numba -c conda-forge -c rapidsai/label/dev -c defaults cudf=0.2.0
```

You can create and activate a development environment using the conda command:

```bash
conda env create --name cudf --file conda_environments/testing_py35.yml
source activate cudf
```

### Pip

Support is coming soon, please use conda for the time being.

## Development Setup for cuDF & libgdf

The following instructions are tested on Linux and OSX systems.

### Get Dependencies for libgdf

Compiler requirement:

* `g++` 5.4
* `cmake` 3.12

CUDA requirement:

* CUDA 9.2+

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Since `cmake` will download and build Apache Arrow (version 0.7.1 or
0.8+) you may need to install Boost C++ (version 1.58+) before running
`cmake`:

```bash
# Install Boost C++ for Ubuntu 16.04/18.04
$ sudo apt-get install libboost-all-dev
```

or

```bash
# Install Boost C++ for Conda
$ conda install -c conda-forge boost
```

### Build from Source

To install cuDF from source, ensure the dependencies are met and follow the steps below:

1. Clone the repository
```bash
git clone --recurse-submodules https://github.com/rapidsai/cudf.git
cd cudf
```
2. Create the conda development environment `cudf` as detailed above
3. Build and install `libgdf`
```bash
source activate cudf
mkdir -p libgdf/build
cd libgdf/build
cmake .. -DHASH_JOIN=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j install
make copy_python
python setup.py install
```
4. Build and install `cudf` from the root of the repository
```bash
cd ../..
python setup.py install
```

## Automated Build in Docker Container

A Dockerfile is provided with a preconfigured conda environment for building and installing the cuDF master branch.

**Notes**:
* We test with and recommended installing [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
* Host's installed nvidia driver must support >= the specified CUDA version (9.2 by default).
* Alternative CUDA_VERSION should be specified via Docker [build-arg](https://docs.docker.com/engine/reference/commandline/build/#set-build-time-variables---build-arg)
* Alternate branches for cudf may be specified as Docker build-args CUDF_REPO. See Dockerfile for example.
* Ubuntu 16.04 is the default OS for this container. Alternate OSes may be specified as Docker build-arg LINUX_VERSION. See list of [available images](https://hub.docker.com/r/nvidia/cuda/).
* Python 3.5 is default, but other versions may be specified via PYTHON_VERSION build-arg
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
/# source activate cudf
(gdf) root@3f689ba9c842:/# python -c "import cudf"
(gdf) root@3f689ba9c842:/#
```

## Testing

### cuDF

This project uses [py.test](https://docs.pytest.org/en/latest/)

In the source root directory and with the development conda environment activated, run:

```bash
py.test --cache-clear --ignore=libgdf
```

### libgdf

The `libgdf` tests require a GPU and CUDA. CUDA can be installed locally or through the conda packages of `numba` & `cudatoolkit`. For more details on the requirements needed to run these tests see the [libgdf README](libgdf/README.md).

`libgdf` has two testing frameworks `py.test` and GoogleTest:

```bash
# Run py.test command inside the /libgdf folder
py.test

# Run GoogleTest command inside the /libgdf/build folder after cmake
make -j test
```

---

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### GPU Arrow

The GPU version of Arrow is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the [Apache Arrow](https://arrow.apache.org/) columnar data format on the GPU. Currently, a subset of the features in Arrow are supported.
