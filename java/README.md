# Java API for cudf

This project provides java bindings for cudf, to be able to process large amounts of data on
a GPU. This is still a work in progress so some APIs may change until the 1.0 release.

## Behavior on Systems with Multiple GPUs

The cudf project currently operates with a single GPU per process.  The CUDA runtime will
automatically search for a GPU to use if a thread has not specifically requested a device,
and Java processes tend to run with many threads.  If one of the Java threads using cudf
does not use the same device being used by other cudf threads then this will lead to an
invalid state that can trigger application crashes.

To avoid these crashes, cudf will remember which CUDA device was used to initialize the
Rapids Memory Manager (RMM) via cudf.  cudf methods will automatically set the thread's
CUDA device to this initial device if necessary.  Note that mixing cudf calls with other
CUDA libraries that could change the thread's current CUDA device will be problematic.

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
If you use the default cmake options libcudart will be dynamically linked to libcudf and librmm
which are included.  If you do this the resulting jar will have a classifier associated with it
because that jar can only be used with a single version of the CUDA runtime.  If you want
to remove that requirement you can build RMM and cuDF with `-DCUDA_STATIC_RUNTIME=ON` when
running cmake, and similarly -DCUDA_STATIC_RUNTIME=ON when running maven.  This will statically 
link in the CUDA runtime and result in a jar with no
classifier that should run on any host that has a version of the driver new enough to support
the runtime that this was built with.  Official releases will indicate in the release notes
the minimum driver version required.

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
