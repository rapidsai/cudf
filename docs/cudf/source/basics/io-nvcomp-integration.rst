nvCOMP Integration
=============================

Some types of compression/decompression can be performed using either `nvCOMP library <https://github.com/NVIDIA/nvcomp>`_ or the internal implementation. 

Which implementation is used by default depends on the data format and the compression type.
Behavior can be influenced through environment variable ``LIBCUDF_NVCOMP_POLICY``.

There are three valid values for the environment variable:

- "STABLE": Only enable the nvCOMP in places where it has been deemed stable for production use. 
- "ALWAYS": Enable all available uses of nvCOMP, including new, experimental combinations.
- "OFF": Disable nvCOMP use whenever possible and use the internal implementations instead.

If no value is set, behavior will be the same as the "STABLE" option.


.. table:: Current policy for nvCOMP use for different types
    :widths: 20 15 15 15 15 15 15 15 15 15

    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
    |                       |       CSV       |      Parquet    |       JSON       |       ORC       |  AVRO  |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
    | Compression Type      | Writer | Reader | Writer | Reader | Writer¹ | Reader | Writer | Reader | Reader |
    +=======================+========+========+========+========+=========+========+========+========+========+
    | snappy                | ❌     | ❌     | Stable | Stable | ❌      | ❌     | Stable | Stable | ❌     |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
