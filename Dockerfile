# An integration test & dev container which builds and installs libgdf & pygdf from master

# Update based on your host's CUDA driver version
FROM nvidia/cuda:9.2-devel

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y \
      git \
      build-essential \
      python3-dev \
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
ENV PYTHON_VERSION=3.6
RUN conda env create -n gdf python=${PYTHON_VERSION} --file /conda_environments/gdf_build.yml

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
ENV CC=/usr/bin/gcc-4.8
ENV CXX=/usr/bin/g++-4.8

# Update the URLs below to build against forks, other branches, or specific PR
RUN git clone --recurse-submodules https://github.com/gpuopenanalytics/libgdf
RUN git clone --recurse-submodules https://github.com/gpuopenanalytics/pygdf
RUN source activate gdf && \
    mkdir -p /libgdf/build && \
    cd /libgdf/build && \
    cmake .. && \
    make install && \
    make copy_python && \
    python setup.py install && \
    cd /pygdf && \
    python setup.py install
