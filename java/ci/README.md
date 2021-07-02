# Build Jar artifact of cuDF

## Build the docker image

### Prerequisite

1. Docker should be installed.
2. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) should be installed.

### Build the docker image

In the root path of cuDF repo, run below command to build the docker image.
```bash
docker build -f java/ci/Dockerfile.centos7 --build-arg CUDA_VERSION=11.2.2 -t cudf-build:11.2.2-devel-centos7 .
```

The following CUDA versions are supported:
* CUDA 11.0
* CUDA 11.1
* CUDA 11.2

Change the --build-arg CUDA_VERSION to what you need.
You can replace the tag "cudf-build:11.2.2-devel-centos7" with another name you like.

## Start the docker then build

### Start the docker

Run below command to start a docker container with GPU.
```bash
nvidia-docker run -it cudf-build:11.2.2-devel-centos7 bash
```

### Download the cuDF source code

You can download the cuDF repo in the docker container or you can mount it into the container.
Here I choose to download again in the container.
```bash
git clone --recursive https://github.com/rapidsai/cudf.git -b branch-21.08
```

### Build cuDF jar with devtoolset

```bash
cd cudf
export WORKSPACE=`pwd`
scl enable devtoolset-9 "java/ci/build-in-docker.sh"
```

### The output

You can find the cuDF jar in java/target/ like cudf-21.08.0-SNAPSHOT-cuda11.jar.

