GPUDirect Storage Integration
=============================

Many IO APIs can use GPUDirect Storage (GDS) library to optimize IO operations. 
GDS enables a direct data path for direct memory access (DMA) transfers between GPU memory and storage, which avoids a bounce buffer through the CPU. 
GDS also has a compatibility mode that allows the library to fall back to copying through a CPU bounce buffer. 
The SDK is available for download `here <https://developer.nvidia.com/gpudirect-storage>`_.
GDS is also included in CUDA Toolkit 11.4 and higher.

Use of GPUDirect Storage in cuDF is enabled by default, but can be disabled through the environment variable ``LIBCUDF_CUFILE_POLICY``. 
This variable also controls the GDS compatibility mode. 

There are three valid values for the environment variable:

- "GDS": Enable GDS use; GDS compatibility mode is *off*.
- "ALWAYS": Enable GDS use; GDS compatibility mode is *on*.
- "OFF": Completely disable GDS use.

If no value is set, behavior will be the same as the "GDS" option.

This environment variable also affects how cuDF treats GDS errors.
When ``LIBCUDF_CUFILE_POLICY`` is set to "GDS" and a GDS API call fails for any reason, cuDF falls back to the internal implementation with bounce buffers.
When ``LIBCUDF_CUFILE_POLICY`` is set to "ALWAYS" and a GDS API call fails for any reason (unlikely, given that the compatibility mode is on), 
cuDF throws an exception to propagate the error to te user.

Operations that support the use of GPUDirect Storage:

- `read_avro`
- `read_parquet`
- `read_orc`
- `to_csv`
- `to_parquet`
- `to_orc`
