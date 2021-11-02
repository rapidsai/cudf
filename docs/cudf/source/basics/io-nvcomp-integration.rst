nvCOMP Integration
=============================

Some types of compression/decompression can be performed using either `nvCOMP library <https://github.com/NVIDIA/nvcomp>`_ or the internal implementation. 

Which implementation is used by default depends on the data format and the compression type. Behavior can be influenced through environment variable ``LIBCUDF_NVCOMP_POLICY``.

There are three special values for the environment variable:

- "STABLE": only enable the nvCOMP readers that have been deemed stable for production use. 
- "EXPERIMENTAL": All available uses of nvCOMP are enabled, including new, experimental combinations.
- "NONE": Internal implementations are used whenever possible.

Any other value (or no value set) will result in the same behavior as the "STABLE" option.


.. table:: Current policy for nvCOMP use for different types
    :widths: 20 15 15 15 15 15 15 15 15 15

    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
    |                       |       CSV       |      Parquet    |       JSON       |       ORC       |  AVRO  |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
    | Compression Type      | Writer | Reader | Writer | Reader | Writer¹ | Reader | Writer | Reader | Reader |
    +=======================+========+========+========+========+=========+========+========+========+========+
    | snappy                | ❌     | ❌     | Stable | Stable | ❌      | ❌     | Stable | Stable | ❌     |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+
