# An integration test & dev container which builds and installs libgdf & pygdf from master

# Specify alternate CUDA toolkit version as Docker build-arg
ARG CUDA_VERSION=9.2
ARG LINUX_VERSION=ubuntu16.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y \
      git \
      gcc-4.8 \
      g++-4.8

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Combined libgdf/pygdf conda env
ADD conda_environments/gdf_build.yml /conda_environments/gdf_build.yml
# Also tested working with 3.5
ARG PYTHON_VERSION=3.6
RUN conda env create -n gdf python=${PYTHON_VERSION} --file /conda_environments/gdf_build.yml

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
ARG CC=/usr/bin/gcc-4.8
ARG CXX=/usr/bin/g++-4.8

# Specify alternate URLs via build-args to build against forks, other branches, or specific PR
ARG LIBGDF_REPO=https://github.com/gpuopenanalytics/libgdf
ARG PYGDF_REPO=https://github.com/gpuopenanalytics/pygdf
# To build container against https://github.com/gpuopenanalytics/pygdf/pull/138:
# docker build --build-arg PYGDF_REPO="https://github.com/dantegd/pygdf -b enh-ext-unique-value-counts" -t gdf .
RUN git clone --recurse-submodules ${LIBGDF_REPO}
RUN git clone --recurse-submodules ${PYGDF_REPO}

RUN source activate gdf && \
    mkdir -p /libgdf/build && \
    cd /libgdf/build && \
    cmake .. && \
    make install && \
    make copy_python && \
    python setup.py install && \
    cd /pygdf && \
    python setup.py install
