## nvstrings

Python wrapper for nvstrings and nvcategory modules.

### API Documentation

The documentation is generated from the source files and available on 
[https://nvstrings.readthedocs.io/en/latest/api.html](https://nvstrings.readthedocs.io/en/latest/api.html)


### Dependencies

Modules only depend on Python 3.6+ and CUDA 9.2+ installed.
There are currently no other python packages required to use or build these modules.

### Build

See the [C/C++ readme](../cpp/README.md) for details on building the python modules.

Build custrings from root of the git repository by:
```
$ ./build.sh custrings
```

### Install

Instructions in [C/C++ readme](../cpp/README.md) also include install the python modules.

### Run

Example of using nvstrings and nvcategory
```
  import nvstrings, nvcategory

  s = nvstrings.to_device(['a','bb','cc','bb','aa'])
  print(s.len())
  [1,2,2,2,2]

  c = nvcategory.from_string(s)
  print(c.keys(), c.values())
  ['a', 'aa', 'bb', 'cc'] [0, 2, 3, 2, 1]

```
