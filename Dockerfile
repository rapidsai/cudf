# An integration test & dev container which builds and installs cuDF from master
ARG CUDA_VERSION=9.2
ARG LINUX_VERSION=ubuntu16.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
# Needed for pygdf.concat(), avoids "OSError: library nvvm not found"
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/

ARG CC=5
ARG CXX=5
RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y \
      git \
      gcc-${CC} \
      g++-${CXX} \
      libboost-all-dev

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Build cuDF conda env
ARG PYTHON_VERSION=3.5
RUN conda create -n cudf python=${PYTHON_VERSION}

ARG NUMBA_VERSION=0.40.0
ARG NUMPY_VERSION=1.14.3
# Locked to Pandas 0.20.3 by https://github.com/rapidsai/cudf/issues/118
ARG PANDAS_VERSION=0.20.3
ARG PYARROW_VERSION=0.10.0
RUN conda install -n cudf -y -c numba -c conda-forge -c defaults \
      numba=${NUMBA_VERSION} \
      numpy=${NUMPY_VERSION} \
      pandas=${PANDAS_VERSION} \
      pyarrow=${PYARROW_VERSION} \
      cmake

# Clone cuDF repo
ARG CUDF_REPO=https://github.com/rapidsai/cudf
ARG CUDF_BRANCH=master
RUN git clone --recurse-submodules -b ${CUDF_BRANCH} ${CUDF_REPO} /cudf

# LibGDF build/install
ENV CC=/usr/bin/gcc-${CC}
ENV CXX=/usr/bin/g++-${CXX}
ARG HASH_JOIN=ON
RUN source activate cudf && \
    mkdir -p /cudf/libgdf/build && \
    cd /cudf/libgdf/build && \
    cmake .. -DHASH_JOIN=${HASH_JOIN} -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
    make -j install && \
    make copy_python && \
    python setup.py install

# cuDF build/install
RUN source activate cudf && \
    cd /cudf && \
    python setup.py install
