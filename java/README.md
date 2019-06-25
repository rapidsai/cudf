# Java API for cudf

This project provides java bindings for cudf, to be able to process large amounts of data on
a GPU. This is still a work in progress so some APIs may change until the 1.0 release.

## Dependency

This is a fat jar with the binary dependencies packaged in the jar.  This means the jar will only
run on platforms the jar was compiled for.  When this is in an official maven repository we will
list the platforms that it is compiled and tested for.  In the mean time you will need to build it
yourself.

For the time being instead of auto-detecting the version of cuda installed on the system at
run-time we use maven classifiers.  The java API remains the same, the difference is the native
runtime it is compiled and linked against.


CUDA 9.2:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <version>${cudf.version}</version>
</dependency>
```

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

Build the native code first, and make sure the a JDK is installed and available. Then run maven
as you would expect.

```
mvn clean install
```

If you have a compatible GPU on your build system the tests will use it.  If not you will see a
lot of skipped tests.