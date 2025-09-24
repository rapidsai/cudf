# Developer Guide

```{note}
At present, this guide only covers the main cuDF library.
In the future, it may be expanded to also cover dask_cudf, cudf_kafka, and custreamz.
```

cuDF is a GPU-accelerated, [Pandas-like](https://pandas.pydata.org/) DataFrame library.
Under the hood, all of cuDF's functionality relies on the CUDA-accelerated `libcudf` C++ library.
Thus, cuDF's internals are designed to efficiently and robustly map pandas APIs to `libcudf` functions.
For more information about the `libcudf` library, a good starting point is the
[developer guide](https://docs.rapids.ai/api/libcudf/stable/developer_guide).

This document assumes familiarity with the
[overall contributing guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md).
The goal of this document is to provide more specific guidance for Python developers.
It covers the structure of the Python code and discusses best practices.
Additionally, it includes longer sections on more specific topics like testing and benchmarking.

```{toctree}
:maxdepth: 2

library_design
contributing_guide
documentation
testing
benchmarking
options
cudf_pandas
udf_memory_management
```
