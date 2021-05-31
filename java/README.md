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
run on platforms the jar was compiled for.  When this is in an official Maven repository we will
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

CUDA 11.0:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <classifier>cuda11</classifier>
    <version>${cudf.version}</version>
</dependency>
```

## Build From Source

Build [libcudf](../cpp) first, and make sure the JDK is installed and available. Specify
the cmake option `-DCUDF_USE_ARROW_STATIC=ON` when building so that Apache Arrow is linked
statically to libcudf, as this will help create a jar that does not require Arrow and its
dependencies to be available in the runtime environment.

After building libcudf, the Java bindings can be built via Maven, e.g.:
```
mvn clean install
```

If you have a compatible GPU on your build system the tests will use it.  If not you will see a
lot of skipped tests.

## Dynamically Linking Arrow

Since libcudf builds by default with a dynamically linked Arrow dependency, it may be
desirable to build the Java bindings without requiring a statically-linked Arrow to avoid
rebuilding an already built libcudf.so. To do so, specify the additional command-line flag
`-DCUDF_JNI_ARROW_STATIC=OFF` when building the Java bindings with Maven.  However this will
result in a jar that requires the correct Arrow version to be available in the runtime
environment, and therefore is not recommended unless you are only performing local testing
within the libcudf build environment.

## Statically Linking the CUDA Runtime

If you use the default cmake options libcudart will be dynamically linked to libcudf
which is included.  If you do this the resulting jar will have a classifier associated with it
because that jar can only be used with a single version of the CUDA runtime.  

There is experimental work to try and remove that requirement but it is not fully functional
you can build cuDF with `-DCUDA_STATIC_RUNTIME=ON` when running cmake, and similarly 
`-DCUDA_STATIC_RUNTIME=ON` when running Maven.  This will statically link in the CUDA runtime
and result in a jar with no classifier that should run on any host that has a version of the
driver new enough to support the runtime that this was built with.

To build the Java bindings with a statically-linked CUDA runtime, use a build command like:
```
mvn clean install -DCUDA_STATIC_RUNTIME=ON
```

You will get errors if the CUDA runtime linking is not consistent.  We tried to detect these
up front and stop the build early if there is a mismatch, but there may be some cases we missed
and this can result in some very hard to debug errors.

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

## GPUDirect Storage (GDS)

The JNI code can be built with *GPUDirect Storage* (GDS) support, which enables direct copying 
between GPU device buffers and supported filesystems (see
https://docs.nvidia.com/gpudirect-storage/).

To enable GDS support, first make sure GDS is installed (see
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html), then run:  
```shell script
cd src/cudf/java
mvn clean install -DUSE_GDS=ON
```
