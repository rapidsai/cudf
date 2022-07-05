# Testing cuDF

## Tooling
Tests in cuDF are written using [`pytest`](https://docs.pytest.org/en/latest/).
Test coverage is measured using [`coverage.py`](https://coverage.readthedocs.io/en/6.4.1/),
specifically the [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) plugin.
Code coverage reports are uploaded to [Codecov](https://app.codecov.io/gh/rapidsai/cudf).
Each PR also indicates whether it increases or decreases in test coverage.

## Test organization

How tests are organized depends on which of the following two groups they fall into:

1. Methods of classes like `DataFrame` or `Series`.
2. Free functions operating on the above classes like `cudf.merge`.

The former should be organized into directories named for the class (all lowercase), e.g. `dataframe/`.

**Open question**: How should we group functionality within a class? How should we organize free functions? Some options:
1. Match the grouping in our docs. It's somewhat arbitrary, but better than nothing.
2. One file per function. This will create _a lot_ of files, but given how many tests we have that might not be crazy.
3. Comparisons to pandas vs other. In some ways this grouping could be helpful, but I suspect it will only exacerbate our current problem of not knowing where to find tests. We need some sort of grouping by functionality.


## Test contents

### Writing tests 

**Open question**: Do the guidelines below apply to pandas comparisons? For pandas comparisons it is IMO far less problematic to include standard and exceptional cases in the same tests.


In general, functionality must be tested both for standard use cases and exceptional cases.
Standard use cases may be covered using parametrization (using `pytest.mark.parametrize`).
These use cases may also use the generic set of fixtures we provide (see **Add link to section**)
Tests of standard use cases should typically include some coverage of:
- Different dtypes, including nested dtypes (especially strings)
- Mixed objects, e.g. binary operations between `DataFrame` and `Series`
- Operations on scalars
- Verifying all combinations of parameters for complex APIs like `cudf.merge`.

Here are some of the most common exceptional cases to test:
1. `Series`/`DataFrame`/`Index` with zero rows
2. `DataFrame` with zero columns
3. All null data
4. For string or list APIs, empty strings/lists
5. For list APIs, lists containing all null elements or empty strings
Most specific APIs will also include a range of other cases.
In general, it is preferable to write separate tests for different exception cases.
Excessive parametrization and branching increases complexity and obfuscates the purpose of a test.

**Open question**: Should we implement something like `benchmark_with_fixture`? If so, what does it cover?


### Parametrization vs. fixtures

Our tests make use of the [`pytest_cases`](https://smarie.github.io/python-pytest-cases/) `pytest` plugin.
This plugin allows us to handle parametrization much more cleanly than pytest does out of the box.
Specifically, it provides some syntactic sugar around

```python
@pytest.mark.parametrize(
    "num", [1, 2, 3]
)
def test_foo(num):
    assert num
```

for when the parameters are nontrivial and require complex initialization.
This is common for tests of functions accepting cuDF objects, such as `cudf.concat`.
With `pytest_cases`, the different cases are instead placed into separate functions and automatically made available.

```python
# test_foo_cases.py
def case_1():
    return 1

def case_2():
    return 2

def case_3():
    return 3

# test_foo.py
@pytest_cases.parametrize_with_cases(num)
def test_foo(num):
    assert num
```

This approach is strongly encouraged within cuDF tests.
It forces developers to put complex initialization into named, documented functions.
That becomes especially valuable when testing APIs with many parameters.
Additionally, cases, like fixtures, are lazily evaluated.
Initializing complex objects inside a `pytest.mark.parametrize` can dramatically slow down test collection,
or even lead to out of memory issues if too many complex cases are collected.
Using lazy case functions ensures that the associated objects are only created on an as-needed basis.


### Testing utility functions

The `cudf.testing` subpackage provides a handful of public utilities for testing the equality of objects.
The internal `cudf.testing._utils` module provides additional helper functions for use in tests.
In particular:
- `testing._utils.assert_eq` is the biggest hammer to reach for. It can be used to compare any objects. **Open question: aside from supporting pandas objects, are any of the other conveniences in this function worth keeping beyond just what's contained in the public functions like `assert_frame_equal`? If not, maybe we should just include a simpler wrapper that handles to/from pandas.**
- For comparing specific objects, use `testing.testing.assert_[frame|series|index]_equal`.
- For verifying that the expected assertions are raised, use `testing._utils.assert_exceptions_equal`.
