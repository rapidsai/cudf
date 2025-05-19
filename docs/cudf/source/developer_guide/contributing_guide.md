# Contributing Guide

This document focuses on a high-level overview of best practices in cuDF.

## Directory structure and file naming

cuDF generally presents the same importable modules and subpackages as pandas.
All Cython code is contained in `python/cudf/cudf/_lib`.

## Code style

cuDF employs a number of linters to ensure consistent style across the code base.
We manage our linters using [`pre-commit`](https://pre-commit.com/).
Developers are strongly recommended to set up `pre-commit` prior to any development.
The `.pre-commit-config.yaml` file at the root of the repo is the primary source of truth linting.
Specifically, cuDF uses the following tools:

- [`ruff`](https://docs.astral.sh/ruff/) checks for general code formatting compliance.
- [`mypy`](http://mypy-lang.org/) performs static type checking.
  In conjunction with [type hints](https://docs.python.org/3/library/typing.html),
  `mypy` can help catch various bugs that are otherwise difficult to find.
- [`codespell`](https://github.com/codespell-project/codespell) finds spelling errors.

Linter config data is stored in a number of files.
We generally use `pyproject.toml` over `setup.cfg` and avoid project-specific files (e.g. `pyproject.toml` > `python/cudf/pyproject.toml`).
However, differences between tools and the different packages in the repo result in the following caveats:

- `isort` must be configured per project to set which project is the "first party" project.

As a result, we currently maintain both root and project-level `pyproject.toml` files.

For more information on how to use pre-commit hooks, see the code formatting section of the
[overall contributing guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#python--pre-commit-hooks).

## Deprecating and removing code

cuDF follows the policy of deprecating code for one release prior to removal.
For example, if we decide to remove an API during the 22.08 release cycle,
it will be marked as deprecated in the 22.08 release and removed in the 22.10 release.
Note that if it is explicitly mentioned in a comment (like `# Do not remove until..`),
do not enforce the deprecation by removing the affected code until the condition in the comment is met.
All internal usage of deprecated APIs in cuDF should be removed when the API is deprecated.
This prevents users from encountering unexpected deprecation warnings when using other (non-deprecated) APIs.
The documentation for the API should also be updated to reflect its deprecation.
When the time comes to remove a deprecated API, make sure to remove all tests and documentation.

Deprecation messages should:
- emit a FutureWarning;
- consist of a single line with no newline characters;
- indicate replacement APIs, if any exist
  (deprecation messages are an opportunity to show users better ways to do things);
- not specify a version when removal will occur (this gives us more flexibility).

For example:
```python
warnings.warn(
    "`Series.foo` is deprecated and will be removed in a future version of cudf. "
    "Use `Series.new_foo` instead.",
    FutureWarning
)
```

```{warning}
Deprecations should be signaled using a `FutureWarning` **not a `DeprecationWarning`**!
`DeprecationWarning` is hidden by default except in code run in the `__main__` module.
```

Deprecations should also be specified in the respective public API docstring using a
`deprecated` admonition:

```
.. deprecated:: 23.08
    `foo` is deprecated and will be removed in a future version of cudf.
```

## `pandas` compatibility

Maintaining compatibility with the [pandas API](https://pandas.pydata.org/docs/reference/index.html) is a primary goal of cuDF.
Developers should always look at pandas APIs when adding a new feature to cuDF.
When introducing a new cuDF API with a pandas analog, we should match pandas as much as possible.
Since we try to maintain compatibility even with various edge cases (such as null handling),
new pandas releases sometimes require changes that break compatibility with old versions.
As a result, our compatibility target is the latest pandas version.

However, there are occasionally good reasons to deviate from pandas behavior.
The most common reasons center around performance.
Some APIs cannot match pandas behavior exactly without incurring exorbitant runtime costs.
Others may require using additional memory, which is always at a premium in GPU workflows.
If you are developing a feature and believe that perfect pandas compatibility is infeasible or undesirable,
you should consult with other members of the team to assess how to proceed.

When such a deviation from pandas behavior is necessary, it should be documented.
For more information on how to do that, see [our documentation on pandas comparison](./documentation.md#comparing-to-pandas).

## Python vs Cython

cuDF makes substantial use of [Cython](https://cython.org/).
Cython is a powerful tool, but it is less user-friendly than pure Python.
It is also more difficult to debug or profile.
Therefore, developers should generally prefer Python code over Cython where possible.

The primary use-case for Cython in cuDF is to expose libcudf C++ APIs to Python.
This Cython usage is generally composed of two parts:
1. A `pxd` file declaring C++ APIs so that they may be used in Cython, and
2. A `pyx` file containing Cython functions that wrap those C++ APIs so that they can be called from Python.

The latter wrappers should generally be kept as thin as possible to minimize Cython usage.
For more information see [our Cython layer design documentation](./library_design.md#the-cython-layer).

In some rare cases we may actually benefit from writing pure Cython code to speed up particular code paths.
Given that most numerical computations in cuDF actually happen in libcudf, however,
such use cases are quite rare.
Any attempt to write pure Cython code for this purpose should be justified with benchmarks.

## Exception handling

In alignment with [maintaining compatibility with pandas](#pandas-compatibility),
any API that cuDF shares with pandas should throw all the same exceptions as the
corresponding pandas API given the same inputs.
However, it is not required to match the corresponding pandas API's exception message.

When writing error messages,
sufficient information should be included to help users locate the source of the error,
such as including the expected argument type versus the actual argument provided.

For parameters that are not yet supported,
raise `NotImplementedError`.
There is no need to mention when the argument will be supported in the future.

### Handling libcudf Exceptions

Standard C++ natively supports various [exception types](https://en.cppreference.com/w/cpp/error/exception),
which Cython maps to [these Python exception types](https://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#exceptions).
In addition to built-in exceptions, libcudf also raises a few additional types of exceptions.
cuDF extends Cython's default mapping to account for these exception types.
When a new libcudf exception type is added, a suitable except clause should be added to cuDF's
[exception handler](https://github.com/rapidsai/cudf/blob/main/python/cudf/cudf/_lib/cpp/exception_handler.hpp).
If no built-in Python exception seems like a good match, a new Python exception should be created.

### Raising warnings

Where appropriate, cuDF should throw the same warnings as pandas.
For instance, API deprecations in cuDF should track pandas as closely as possible.
However, not all pandas warnings are appropriate for cuDF.
Common exceptional cases include
[implementation-dependent performance warnings](https://pandas.pydata.org/docs/reference/api/pandas.errors.PerformanceWarning.html)
that do not apply to cuDF's internals.
The decision of whether or not to match pandas must be made on a case-by-case
basis and is left to the best judgment of developers and PR reviewers.

### Catching warnings from dependencies

cuDF developers should avoid using deprecated APIs from package dependencies.
However, there may be cases where such uses cannot be avoided, at least in the short term.
If such cases arise, developers should
[catch the warnings](https://docs.python.org/3/library/warnings.html#warnings.catch_warnings)
within cuDF so that they are not visible to the user.
