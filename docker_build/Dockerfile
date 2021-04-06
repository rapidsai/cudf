# Copyright (c) 2021, NVIDIA CORPORATION.

ARG CUDA_VERSION=11.2.2
FROM nvidia/cuda:${CUDA_VERSION}-devel
ENV CUDA_SHORT_VERSION=11.2

SHELL ["/bin/bash", "-c"]
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin

# Build env variables for arrow
ENV CMAKE_BUILD_TYPE=release
ENV PYARROW_WITH_PARQUET=1
ENV PYARROW_WITH_CUDA=1
ENV PYARROW_WITH_ORC=1
ENV PYARROW_WITH_DATASET=1

ENV ARROW_HOME=/repos/dist

# Build env variables for rmm
ENV INSTALL_PREFIX=/usr


RUN apt update -y --fix-missing && \
    apt upgrade -y && \
      apt install -y --no-install-recommends software-properties-common && \
      add-apt-repository ppa:deadsnakes/ppa && \
      apt update -y --fix-missing

RUN apt install -y --no-install-recommends \
      git \
      libboost-all-dev \
      python3.8-dev \
      build-essential \
      autoconf \
      bison \
      flex \
      libboost-filesystem-dev \
      libboost-system-dev \
      libboost-regex-dev \
      libjemalloc-dev \
      wget \
      libssl-dev \
      protobuf-compiler && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py

# Install cmake
RUN version=3.18 && build=5 && mkdir ~/temp && cd ~/temp && wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && cd cmake-$version.$build/ && ./bootstrap && make -j$(nproc) && make install

# Install arrow from source
RUN git clone https://github.com/apache/arrow.git /repos/arrow && mkdir /repos/dist/ && cd /repos/arrow && git checkout apache-arrow-1.0.1 && git submodule init && \
    git submodule update && export PARQUET_TEST_DATA="${PWD}/cpp/submodules/parquet-testing/data" && export ARROW_TEST_DATA="${PWD}/testing/data" && \
    cd /repos/arrow/cpp && mkdir release && cd /repos/arrow/cpp/release && pip install -r /repos/arrow/python/requirements-build.txt && \
    cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME -DCMAKE_INSTALL_LIBDIR=lib -DARROW_FLIGHT=ON -DARROW_GANDIVA=OFF -DARROW_ORC=ON -DARROW_WITH_BZ2=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_BROTLI=ON -DARROW_PARQUET=ON -DARROW_PYTHON=ON -DARROW_PLASMA=ON -DARROW_BUILD_TESTS=ON -DARROW_CUDA=ON -DARROW_DATASET=ON .. && \
    make -j$(nproc) && make install && cd /repos/arrow/python/ && python setup.py build_ext --build-type=release bdist_wheel && pip install /repos/arrow/python/dist/*.whl


# Install rmm from source
RUN cd /repos/ && git clone https://github.com/rapidsai/rmm.git && cd /repos/rmm/ && ./build.sh librmm && pip install /repos/rmm/python/.

ADD . /repos/cudf/

# Build env for CUDF build
ENV CUDF_HOME=/repos/cudf/
ENV CUDF_ROOT=/repos/cudf/cpp/build/

# Install cudf from source
RUN cd /repos/cudf/ && git submodule update --init --recursive && ./build.sh libcudf && \
    pip install /repos/cudf/python/cudf/.

