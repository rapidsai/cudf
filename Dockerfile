# An integration test & dev container which builds and installs libgdf & pygdf from master
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
      libboost-dev \
      cmake

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Build combined libgdf/pygdf conda env
ARG PYTHON_VERSION=3.6
RUN conda create -n gdf python=${PYTHON_VERSION}
RUN conda install -n gdf -y -c numba -c conda-forge -c defaults \
      numba \
      pandas

# LibGDF build/install
ARG LIBGDF_REPO=https://github.com/gpuopenanalytics/libgdf
RUN git clone --recurse-submodules ${LIBGDF_REPO} /libgdf
ENV CC=/usr/bin/gcc-${CC}
ENV CXX=/usr/bin/g++-${CXX}
ARG HASH_JOIN=ON
RUN source activate gdf && \
    mkdir -p /libgdf/build && \
    cd /libgdf/build && \
    cmake .. -DHASH_JOIN=${HASH_JOIN} && \
    make -j install && \
    make copy_python && \
    python setup.py install

# PyGDF build/install
ARG PYGDF_REPO=https://github.com/gpuopenanalytics/pygdf
# To build container against https://github.com/gpuopenanalytics/pygdf/pull/138:
# docker build --build-arg PYGDF_REPO="https://github.com/dantegd/pygdf -b enh-ext-unique-value-counts" -t gdf .
RUN git clone --recurse-submodules ${PYGDF_REPO} /pygdf
RUN source activate gdf && \
    cd /pygdf && \
    python setup.py install
