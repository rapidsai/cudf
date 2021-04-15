# Copyright (c) 2021, NVIDIA CORPORATION.

# An integration test & dev container which builds and installs cuDF from main
ARG CUDA_VERSION=11.0
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
ENV DEBIAN_FRONTEND=noninteractive

ARG CC=9
ARG CXX=9
RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update -y --fix-missing && \
    apt install -y --no-install-recommends \
      git \
      gcc-${CC} \
      g++-${CXX} \
      libboost-all-dev \
      tzdata && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Build cuDF conda env
ARG CUDA_SHORT_VERSION
ARG PYTHON_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION
ARG NUMBA_VERSION
ENV NUMBA_VERSION=$NUMBA_VERSION
ARG NUMPY_VERSION
ENV NUMPY_VERSION=$NUMPY_VERSION
ARG PANDAS_VERSION
ENV PANDAS_VERSION=$PANDAS_VERSION
ARG PYARROW_VERSION
ENV PYARROW_VERSION=$PYARROW_VERSION
ARG CYTHON_VERSION
ENV CYTHON_VERSION=$CYTHON_VERSION
ARG CMAKE_VERSION
ENV CMAKE_VERSION=$CMAKE_VERSION
ARG CUDF_REPO=https://github.com/rapidsai/cudf
ENV CUDF_REPO=$CUDF_REPO
ARG CUDF_BRANCH=main
ENV CUDF_BRANCH=$CUDF_BRANCH

# Add everything from the local build context
ADD . /cudf/

# Checks if local build context has the source, if not clone it then run a bash script to modify
# the environment file based on versions set in build args
RUN ls -la /cudf
RUN if [ -f /cudf/docker/package_versions.sh ]; \
    then /cudf/docker/package_versions.sh /cudf/conda/environments/cudf_dev_cuda${CUDA_SHORT_VERSION}.yml && \
         conda env create --name cudf --file /cudf/conda/environments/cudf_dev_cuda${CUDA_SHORT_VERSION}.yml ; \
    else rm -rf /cudf && \
         git clone --recurse-submodules -b ${CUDF_BRANCH} ${CUDF_REPO} /cudf && \
         /cudf/docker/package_versions.sh /cudf/conda/environments/cudf_dev_cuda${CUDA_SHORT_VERSION}.yml && \
         conda env create --name cudf --file /cudf/conda/environments/cudf_dev_cuda${CUDA_SHORT_VERSION}.yml ; \
    fi

ENV CC=/usr/bin/gcc-${CC}
ENV CXX=/usr/bin/g++-${CXX}

# libcudf & cudf build/install
RUN source activate cudf && \
    cd /cudf/ && \
    ./build.sh libcudf cudf
