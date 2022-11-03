# Testing cuDF

## Tooling
Tests in cuDF are written using [`pytest`](https://docs.pytest.org/en/latest/).
Test coverage is measured using [`coverage.py`](https://coverage.readthedocs.io/en/latest/),
specifically the [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) plugin.
Code coverage reports are uploaded to [Codecov](https://app.codecov.io/gh/rapidsai/cudf).
Each PR also indicates whether it increases or decreases test coverage.

## Test organization

How tests are organized depends on which of the following two groups they fall into:

1. Free functions such as `cudf.merge` that operate on classes like `DataFrame` or `Series`.
2. Methods of the above classes.

Tests of free functions should be grouped into files based on the
[API sections in the documentation](https://docs.rapids.ai/api/cudf/latest/api_docs/index.html).
This places tests of similar functionality in the same module.
Tests of class methods should be organized in the same way, except that this organization should be within a subdirectory corresponding to the class.
For instance, tests of `DataFrame` indexing should be placed into `dataframe/test_indexing.py`.
In cases where tests may be shared by multiple classes sharing a common parent (e.g. `DataFrame` and `Series` both require `IndexedFrame` tests),
the tests may be placed in a directory corresponding to the parent class.

## Test contents

### Writing tests

In general, functionality must be tested for both standard and exceptional cases.
Standard use cases may be covered using parametrization (using `pytest.mark.parametrize`).
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
6. For numeric data:
  1. All 0s.
  2. All 1s.
  3. Containing/all inf
  4. Containing/all nan
  5. `INT${PRECISION}_MAX` for a given precision (e.g. `2**32` for `int32`).

Most specific APIs will also include a range of other cases.

In general, it is preferable to write separate tests for different exceptional cases.
Excessive parametrization and branching increases complexity and obfuscates the purpose of a test.
Typically, exception cases require specific assertions or other special logic, so they are best kept separate.
The main exception to this rule is tests based on comparison to pandas.
Such tests may test exceptional cases alongside more typical cases since the logic is generally identical.

### Parametrization: custom fixtures and `pytest.mark.parametrize`

When it comes to parametrizing tests written with `pytest`,
the two main options are [fixtures](https://docs.pytest.org/en/latest/explanation/fixtures.html)
and [`mark.parametrize`](https://docs.pytest.org/en/latest/how-to/parametrize.html#pytest-mark-parametrize).
By virtue of being functions, fixtures are both more verbose and more self-documenting.
Fixtures also have the significant benefit of being constructed lazily,
whereas parametrizations are constructed at test collection time.

In general, these approaches are applicable to parametrizations of different complexity.
For the purpose of this discussion,
we define a parametrization as "simple" if it is composed of a list (possibly nested) of primitive objects.
Examples include a list of integers or a list of list of strings.
This _does not_ include e.g. cuDF or pandas objects.
In particular, developers should avoid performing GPU memory allocations during test collection.

With that in mind, here are some ground rules for how to parametrize.

Use `pytest.mark.parametrize` when:
- One test must be run on many inputs and those inputs are simple to construct.

Use fixtures when:
- One or more tests must be run on the same set of inputs,
  and all of those inputs can be constructed with simple parametrizations.
  In practice, that means that it is acceptable to use a fixture like this:
  ```python
      @pytest.fixture(params=["a", "b"])
      def foo(request):
          if request.param == "a":
              # Some complex initialization
          elif request.param == "b":
              # Some other complex initialization
  ```
  In other words, the construction of the fixture may be complex,
  as long as the parametrization of that construction is simple.
- One or more tests must be run on the same set of inputs,
  and at least one of those inputs requires complex parametrizations.
  In this case, the parametrization of a fixture should be decomposed
  by using fixtures that depend on other fixtures.
  ```python
      @pytest.fixture(params=["a", "b"])
      def foo(request):
          if request.param == "a":
              # Some complex initialization
          elif request.param == "b":
              # Some other complex initialization

      @pytest.fixture
      def bar(foo):
         # do something with foo like initialize a cudf object.

      def test_some_property(bar):
          # will be run for each value of bar that results from each value of foo.
          assert some_property_of(bar)
  ```

#### Complex parametrizations

The lists above document common use cases.
However, more complex cases may arise.
One of the most common alternatives is where, given a set of test cases,
different tests need to run on different subsets with a nonempty intersection.
Fixtures and parametrization are only capable of handling the Cartesian product of parameters,
i.e. "run this test for all values of `a` and all values of `b`".

There are multiple potential solutions to this problem.
One possibility is to encapsulate common test logic in a helper function,
then call it from multiple `test_*` functions that construct the necessary inputs.
Another possibility is to use functions rather than fixtures to construct inputs, allowing for more flexible input construction:
```python
def get_values(predicate):
    values = range(10)
    yield from filter(predicate, values)

def test_evens():
    for v in get_values(lambda x: x % 2 == 0):
        # Execute test

def test_odds():
    for v in get_values(lambda x: x % 2 == 1):
        # Execute test
```

Other approaches are also possible, and the best solution should be discussed on a case-by-case basis during PR review.

### Testing utility functions

The `cudf.testing` subpackage provides a handful of utilities for testing the equality of objects.
The internal `cudf.testing._utils` module provides additional helper functions for use in tests.
In particular:
- `testing._utils.assert_eq` is the biggest hammer to reach for. It can be used to compare any pair of objects.
- For comparing specific objects, use `testing.testing.assert_[frame|series|index]_equal`.
- For verifying that the expected assertions are raised, use `testing._utils.assert_exceptions_equal`.
