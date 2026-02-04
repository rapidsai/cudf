# Standalone libcudf C++ Application Demonstrating Use of the Pack and Unpack APIs

This C++ example demonstrates the use of libcudf to pack and unpack table data to and
from device and host memory.

There are three examples included:
1. `device_pack_unpack.cpp`
   This example creates a simple cuDF table on the device and then packs and unpacks
   the table on the device. The original and unpacked tables are printed to show
   no change in content.

2. `host_pack_unpack.cpp`
   This example creates a simple cuDF table on the device and then packs the table
   on the host. The table is then unpacked on the host for lazy access by the device.
   The original and unpacked tables are printed to show no change in content.

3. `host_pack_copy_unpack.cpp`
   This example creates a simple cuDF table on the device and then packs the table
   on the host. The packed table is then copied to a different address on the host
   (simulating host to host copy) and later unpacked on the host for lazy access by
   the device. The original and unpacked tables are printed to show no change in content.

## Compile and Execute

```bash
# Configure project
cmake -S . -B build/

# Build
cmake --build build/ --parallel $PARALLEL_LEVEL

# Execute
build/device_pack_unpack_example
build/host_pack_unpack_example
build/host_pack_copy_unpack_example
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
