# cuDF Error Handling Documentation (draft)

## Scope

This document serves to guide the use of error types when working with exceptions encountered within cuDF domain.
As of the draft version,
it also serves as a reference for supporting libraries (like libcudf) to boost awareness of existing cuDF error types,
and to standardize the error interface.
Note that this document will not serve as a complete list of error types thrown by cuDF. (Maybe it will?)

## Overview

On a high level,
exceptions consist of exception type and exception payload.
cuDF only uses excpetion type to indicate the type of error,
the payload is used to provide additional information for trouble shooting.

cuDF expects to use a mixture of python [builtin error types](https://docs.python.org/3/library/exceptions.html) and custom error types to identify errors.
This is analogous to pandas where custom error types,
deriving from built-in error types are provided. For example: [DuplicateLabelErrors](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.errors.DuplicateLabelError.html).

Examples use of builtin error types are shown below:

| builtin error type | semantic | example use case |
| ------------------ | -------- | ---------------- |
|     TypeError      | Operation is applied to unsupported objects. For objects that is not **yet** supported, use `NotImplementedError` | [binary operation on two unsupported column types](https://github.com/rapidsai/cudf/blob/7d49f75df9681dbe1653029e7d508355884a6d86/python/cudf/cudf/core/column/numerical.py#L110) |
|     ValueError     | The value of the object does not satisfy operation requirement. Usually raising this error requires data introspection. In cases where the object only accept from a finite set of values (such as enum or string set), also use `ValueError`. | [The dataframe to describe is empty](https://github.com/rapidsai/cudf/blob/7d49f75df9681dbe1653029e7d508355884a6d86/python/cudf/cudf/core/dataframe.py#L5366) |
|     IndexError     | Array access out of range | [Retrieving rows from a column specified by an out of bound index](https://github.com/rapidsai/cudf/blob/7d49f75df9681dbe1653029e7d508355884a6d86/python/cudf/cudf/core/column/column.py#L849-L851) |
|     KeyError       | Mapping access with invalid key | [Retrieving rows from column specified by an invalid key](https://github.com/rapidsai/cudf/blob/7d49f75df9681dbe1653029e7d508355884a6d86/python/cudf/cudf/core/indexing.py#L177) |
| NotImplementedError| Operation of object is planned, but not yet supported | (none) |

|  FileNotFoundError | File could not be found at user specified location | https://github.com/rapidsai/cudf/blob/4893259b2fd6cf1f2079eff68249290708519892/python/cudf/cudf/_lib/csv.pyx#L369 |
|   OverflowError    | Numerical overflows | (Usually surfacing from supporting libraries?)  |
|     IOError        | (TBA by cuIO) | (TBA) |
|     OSError        | Operating system failure | (TBA) |

Examples of custom builtin error types:
(TBA)

Errors raised by the API are considered part of the protocol and should be documented in the docstring.
For a specific invalid input, changing the raised exception is considered a breaking change.

## Handling Exceptions Thrown by Supporting Libraries

cuDF depends on external libraries to function.
While cuDF does extensive tests on inputs to make sure data passed to lower level is legal,
supporting libraries may also perform their own data checks.
Very often such checks are redundant.
Double-checking adds unecessary latency to cuDF API calls and is thus unwanted.
To allow reusing error checks from supporting libraries,
the following section explains how cuDF proposes to map errors from supporting libraries to cuDF errors.

Note that this does not mean cuDF will skip checking errors.
As supporting libraries have different performance constraints to cuDF,
cuDF should perform checks when necessary.
An example is that libcudf [does not introspect data](https://github.com/rapidsai/cudf/issues/5505),
but cuDF should to make sure the data is valid.

### Python Libraries

Exceptions from upstream python libraries can be interpreted in many ways.
In general,
cuDF should do argument validation and only calls the API with valid inputs.
If certain error returns from upstream libraries and is confusing to user,
it should be treated as a corner case bug and the input should be handled correctly before passing upstream.

In situations where cuDF doesn't have control over input (e.g. I/O, resource management issue),
cuDF can surface the error as is if it's not too confusing for users.
Otherwise,
develoepr should create a custom error type wrapper over the upstream error type to provide more context of the error.

### C++ Libraries

cuDF interfaces with C++ libraries through Cython.
By default,
cuDF captures all exceptions thrown by C++ libraries.
Cython maps these exceptions into Python errors as outlined by [this document](http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions).
Besides,
it is also possible to create custom c++ error types and map it to a custom python error type, as shown in [this page](https://stackoverflow.com/questions/10684983/handling-custom-c-exceptions-in-cython).

Each error type rasied from c++ should be unambiguously documented and mapped to a python error type,
cuDF will surface such exceptions via Cython's C++->Python exception mapping.

The contents of [`what()`](https://www.cplusplus.com/reference/exception/exception/what/) should only be used to provide additional context of the error and will not be used to determine the type of error.
