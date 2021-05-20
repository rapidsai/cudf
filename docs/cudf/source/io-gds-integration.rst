GPUDirect Storage Integration
=============================

Many IO APIs can use GPUDirect Storage (GDS) library to optimize IO operations. 
GDS enables a direct data path for direct memory access (DMA) transfers between GPU memory and storage, which avoids a bounce buffer through the CPU. 
GDS also has a compatibility mode that allows the library to fall back to copying through a CPU bounce buffer. 
The SDK is available for download `here <https://developer.nvidia.com/gpudirect-storage>`_.

Use of GPUDirect Storage in cuDF is disabled by default, and can be enabled through environment variable ``LIBCUDF_CUFILE_POLICY``. 
This variable also controls the GDS compatibility mode. There are two special values for the environment variable:

- "GDS": Use of GDS is enabled; GDS compatibility mode is *off*.
- "ALWAYS": Use of GDS is enabled; GDS compatibility mode is *on*.

Any other value (or no value set) will keep the GDS disabled for use in cuDF and IO will be done using cuDF's CPU bounce buffers.

This environment variable also affects how cuDF treats GDS errors.
When ``LIBCUDF_CUFILE_POLICY`` is set to "GDS" and a GDS API call fails for any reason, cuDF falls back to the internal implementation with bounce buffers.
When ``LIBCUDF_CUFILE_POLICY`` is set to "ALWAYS" and a GDS API call fails for any reason (unlikely, given that the compatibility mode is on), 
cuDF throws an exception to propagate the error to te user.

NOTE: current GDS integration is not fully optimized and enabling GDS will not lead to performance improvements in all cases.