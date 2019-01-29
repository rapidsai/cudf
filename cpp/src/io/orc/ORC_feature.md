# Quick Guide for ORC feature

ORC is a file format named Optimized Row Columnar from [https://orc.apache.org].
The goal of this library is to accelarate ORC file reading by GPU.
However, the purpose for current version is still feature test, the performance is not good yet.


## compression type

None and zlib are supported (CompressionKind::NONE, ZLIB).
However, zlib is decoded by `CPU` at this point. It is still slow. Decoding by GPU is a future task.

Other compression types are not supported (snappy, lzo, lz4, zstd).

## Data Types

These data types are supported unless they are not a member of list/map/union.

* boolean
* bytes
* integer (tinyint, smallint, int, bigint)
* float
* double
* date
* timestamp
* string (string, char, varchar, binary)


These data types are not supported because cuDF does not support them yet.

* list
* map
* union
* decimal


## example of file reading

src/tests/io/orc/gdf_interface_*.cpp files would be the examples and testings for orc file reading.



-------------------------------------------------------------------------------
# Software Requirement

Library requirement (will be installed by conda):

* `protobuf` 3.6.0
* `zlib` 1.2.8 - 1.3

CUDA requirement:

* CUDA 9.0+

Thirdparty library (git submodule)

* apatch orc 
   ORC schema file (orc_proto.proto) and example files are only referenced.


