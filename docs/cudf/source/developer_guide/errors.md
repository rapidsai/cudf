# cuDF Error Handling Documentation

This document discusses what to raise given invalid user inputs and how to handle libcudf exceptions in cuDF.

## Overview

cuDF uses [builtin error types](https://docs.python.org/3/library/exceptions.html) to indicate user error.
If an API shares an argument with pandas,
cuDF should raise the same error type given a specific invalid input for that argument.
Developers of the API should include sufficient information in the exception message to help user locate the source of the error,
but it is not required to match the corresponding pandas API's exception message.

For parameters that are not yet supported,
raise [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError).

## Handling libcudf Exceptions

Currently libcudf raises `cudf::logic_error` and `cudf::cuda_error`.
By default these error types are mapped to `RuntimeError` in python.
The `what()` message should be ignored in python level and not used to determine the error type raised from libcudf. 

The projected roadmap for libcudf is to diversify the exceptions types on different invalid inputs.
Cython maps all standard exception types to python error types,
as shown in [this document](http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions).
In case it is insufficient to use only standard exception types,
derived exception types may be created.
On the python level custom error type should be created accordingly.
