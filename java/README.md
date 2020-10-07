# Java API for cudf

This project provides java bindings for cudf, to be able to process large amounts of data on
a GPU. This is still a work in progress so some APIs may change until the 1.0 release.

## Behavior on Systems with Multiple GPUs

The cudf project currently works with a single GPU per process. The CUDA runtime
assigns the default GPU to all new operating system threads when they start to
interact with CUDA. This means that if you use a multi-threaded environment,
like Java processes tend to be, and try to use a non-default GPU for cudf you
can run into hard to debug issues including resource leaks, invalid states,
application crashes, and poor performance.

To prevent this the Java cudf API will remember the device used to initialize
the Rapids Memory Manager (RMM), and automatically set the thread's active
device to it, if needed. It will not set the device back when the cudf call
completes. This is different from most CUDA libraries and can result in
unexpected behavior if you try to mix these libraries using the same thread.

## Dependency

This is a fat jar with the binary dependencies packaged in the jar.  This means the jar will only
run on platforms the jar was compiled for.  When this is in an official maven repository we will
list the platforms that it is compiled and tested for.  In the mean time you will need to build it
yourself. In official releases there should be no classifier on the jar and it should run against
most modern cuda drivers.

```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <version>${cudf.version}</version>
</dependency>
```

In some cases there may be a classifier to indicate the version of cuda required. See the 
Build From Source section below for more information about when this can happen. No official
release of the jar will have a classifier on it.

CUDA 10.0:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <classifier>cuda10</classifier>
    <version>${cudf.version}</version>
</dependency>
```

## Build From Source

Build the native code first, and make sure the a JDK is installed and available.

When building libcudf, make sure you install boost first:
```bash
# Install Boost C++ for Ubuntu 16.04/18.04/20.04
sudo apt install libboost-filesystem-dev
```
or for a smaller installation footprint (Boost is a large library), build it from the source:
```bash
wget https://dl.bintray.com/boostorg/release/1.74.0/source/boost_1_74_0.tar.bz2
tar xvf boost_1_74_0.tar.bz2
cd boost_1_74_0
./bootstrap.sh --with-libraries=filesystem
./b2 cxxflags=-fPIC link=static
sudo cp stage/lib/libboost_filesystem.a /usr/local/lib/
```
and pass in the cmake options
`-DARROW_STATIC_LIB=ON -DBoost_USE_STATIC_LIBS=ON` so that Apache Arrow and Boost libraries are
linked statically.

If you use the default cmake options libcudart will be dynamically linked to libcudf
which is included.  If you do this the resulting jar will have a classifier associated with it
because that jar can only be used with a single version of the CUDA runtime.  

There is experimental work to try and remove that requirement but it is not fully functional
you can build cuDF with `-DCUDA_STATIC_RUNTIME=ON` when running cmake, and similarly 
`-DCUDA_STATIC_RUNTIME=ON` when running maven.  This will statically link in the CUDA runtime
and result in a jar with no classifier that should run on any host that has a version of the
driver new enough to support the runtime that this was built with. Unfortunately `libnvrtc` is still
required for runtime code generation which also is tied to a specific version of cuda.

To build with maven for dynamic linking you would run.

```
mvn clean install
```

for static linking you would run

```
mvn clean install -DCUDA_STATIC_RUNTIME=ON
```

You will get errors if you don't do it consistently.  We tried to detect these up front and stop the build early if there is a mismatch, but there may be some cases we missed and this can result in some very hard to debug errors.

If you have a compatible GPU on your build system the tests will use it.  If not you will see a
lot of skipped tests.

## Per-thread Default Stream

The JNI code can be built with *per-thread default stream* (PTDS), which gives each host thread its
own default CUDA stream, and can potentially increase the overlap of data copying and compute
between different threads (see
[blog post](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)).

Since the PTDS option is for each compilation unit, it should be done at the same time across the
whole codebase. To enable PTDS, first build cuDF:
```shell script
cd src/cudf/cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DPER_THREAD_DEFAULT_STREAM=ON
make -j`nproc`
make install
```

then build the jar:
```shell script
cd src/cudf/java
mvn clean install -DPER_THREAD_DEFAULT_STREAM=ON
```
