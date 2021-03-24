FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
ARG PARALLEL_LEVEL=8

# Install basic cudf dependencies
RUN GCC_VERSION=9 \
 && apt update -y \
 && apt install -y software-properties-common \
 && add-apt-repository -y ppa:git-core/ppa \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt install -y \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    git nano sudo wget ninja-build bash-completion \
    # ccache dependencies
    unzip automake autoconf libb2-dev libzstd-dev \
    # CMake dependencies
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
    # cuDF dependencies
    libboost-filesystem-dev \
 && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 # Remove any existing gcc and g++ alternatives
 && update-alternatives --remove-all cc  >/dev/null 2>&1 || true \
 && update-alternatives --remove-all c++ >/dev/null 2>&1 || true \
 && update-alternatives --remove-all gcc >/dev/null 2>&1 || true \
 && update-alternatives --remove-all g++ >/dev/null 2>&1 || true \
 && update-alternatives --remove-all gcov >/dev/null 2>&1 || true \
 && update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    --slave /usr/bin/cc cc /usr/bin/gcc-${GCC_VERSION} \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/c++ c++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION} \
 # Set gcc-${GCC_VERSION} as the default gcc
 && update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION}

ARG CMAKE_VERSION=3.18.5

# Install CMake
RUN cd /tmp \
 && curl -fsSLO --compressed "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" -o /tmp/cmake-$CMAKE_VERSION.tar.gz \
 && tar -xvzf /tmp/cmake-$CMAKE_VERSION.tar.gz && cd /tmp/cmake-$CMAKE_VERSION \
 && /tmp/cmake-$CMAKE_VERSION/bootstrap \
    --system-curl \
    --parallel=$PARALLEL_LEVEL \
 && make install -j$PARALLEL_LEVEL \
 && cd /tmp && rm -rf /tmp/cmake-$CMAKE_VERSION*

ARG CCACHE_VERSION=4.1

# Install ccache
RUN cd /tmp \
 && curl -fsSLO --compressed https://github.com/ccache/ccache/releases/download/v$CCACHE_VERSION/ccache-$CCACHE_VERSION.tar.gz -o /tmp/ccache-$CCACHE_VERSION.tar.gz \
 && tar -xvzf /tmp/ccache-$CCACHE_VERSION.tar.gz && cd /tmp/ccache-$CCACHE_VERSION \
 && mkdir -p /tmp/ccache-$CCACHE_VERSION/build \
 && cd /tmp/ccache-$CCACHE_VERSION/build \
 && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DZSTD_FROM_INTERNET=ON \
    -DENABLE_TESTING=OFF \
    /tmp/ccache-$CCACHE_VERSION \
 && make install -j$PARALLEL_LEVEL \
 && cd /tmp && rm -rf /tmp/ccache-$CCACHE_VERSION*

ENV CUDA_HOME="/usr/local/cuda"

RUN mkdir -p /workspace
WORKDIR /workspace
