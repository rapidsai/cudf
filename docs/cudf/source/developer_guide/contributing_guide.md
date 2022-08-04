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

- [`flake8`](https://github.com/pycqa/flake8) checks for general code formatting compliance. 
- [`black`](https://github.com/psf/black) is an automatic code formatter.
- [`isort`](https://pycqa.github.io/isort/) ensures imports are sorted consistently.
- [`mypy`](http://mypy-lang.org/) performs static type checking.
  In conjunction with [type hints](https://docs.python.org/3/library/typing.html),
  `mypy` can help catch various bugs that are otherwise difficult to find.
- [`pydocstyle`](https://github.com/PyCQA/pydocstyle/) lints docstring style.

Linter config data is stored in a number of files.
We generally use `pyproject.toml` over `setup.cfg` and avoid project-specific files (e.g. `setup.cfg` > `python/cudf/setup.cfg`).
However, differences between tools and the different packages in the repo result in the following caveats:

- `flake8` has no plans to support `pyproject.toml`, so it must live in `setup.cfg`.
- `isort` must be configured per project to set which project is the "first party" project.

Additionally, our use of `versioneer` means that each project must have a `setup.cfg`.
As a result, we currently maintain both root and project-level `pyproject.toml` and `setup.cfg` files.

For more information on how to use pre-commit hooks, see the code formatting section of the
[overall contributing guide](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#python--pre-commit-hooks).

## Deprecating and removing code

cuDF follows the policy of deprecating code for one release prior to removal.
For example, if we decide to remove an API during the 22.08 release cycle,
it will be marked as deprecated in the 22.08 release and removed in the 22.10 release.
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

This section is under development, see https://github.com/rapidsai/cudf/pull/7917.
