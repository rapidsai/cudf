# Testing cuDF

## Tooling
Tests in cuDF are written using [`pytest`](https://docs.pytest.org/en/latest/).
Test coverage is measured using [`coverage.py`](https://coverage.readthedocs.io/en/latest/),
specifically the [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) plugin.
Code coverage reports are uploaded to [Codecov](https://app.codecov.io/gh/rapidsai/cudf).
Each PR also indicates whether it increases or decreases test coverage.

### Configuring pytest

Pytest will accept configuration in [multiple different
files](https://docs.pytest.org/en/stable/reference/customize.html),
with a specified discovery and precedence order. Note in particular
that there is no automatic "include" mechanism, as soon as a matching
configuration file is found, discovery stops.

For preference, so that all tool configuration lives in the same
place, we use `pyproject.toml`-based configuration. Test configuration
for a given package should live in that package's `pyproject.toml`
file.

Where tests do not naturally belong to a project, for example the
`cudf.pandas` integration tests and the cuDF benchmarks, use a
`pytest.ini` file as close to the tests as possible.

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

(test_parametrization)=

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

(xfailing_tests)=

### Tests with expected failures (`xfail`s)

In some circumstances it makes sense to mark a test as _expected_ to
fail, perhaps because the functionality is not yet implemented in
cuDF. To do so use the
[`pytest.mark.xfail`](https://docs.pytest.org/en/stable/reference/reference.html#pytest.mark.xfail)
fixture on the test.

If the test is parametrized and only a single parameter is expected to
fail, rather than marking the entire test as xfailing, mark the single
parameter by creating a
[`pytest.param`](https://docs.pytest.org/en/stable/how-to/skipping.html#skip-xfail-with-parametrize)
with appropriate marks.

```python
@pytest.mark.parametrize(
    "value",
    [
        1,
        2,
        pytest.param(
            3, marks=pytest.mark.xfail(reason="code doesn't work for 3")
        ),
    ],
)
def test_value(value):
    assert value < 3
```

When marking an `xfail`ing test, provide a descriptive reason. This
_should_ include a link to an issue describing the problem so that
progress towards fixing the problem can be tracked. If no such issue
exists already, create one!

#### Conditional `xfail`s

Sometimes, a parametrized test is only expected to fail for some
combination of its parameters. Say, for example, division by zero but
only if the datatype is `bool`. If all combinations with a given
parameter are expected to fail, one can mark the parameter with
`pytest.mark.xfail`, indicating a reason for the expected failure. If
only _some_ of the combinations are expected to fail, it can be
tempting to mark the parameter as `xfail`, but this should be avoided.
A test marked as `xfail` that passes is an "unexpected pass" or
`XPASS` which is considered a failure by the test suite since we use
the pytest option
[`xfail_strict=true`](https://docs.pytest.org/en/latest/how-to/skipping.html#strict-parameter).
Another option is to use the programmatic `pytest.xfail` function to
fail in the test body to `xfail` the relevant combination of
parameters. **DO NOT USE THIS OPTION**. Unlike the mark-based
approach, `pytest.xfail` *does not* run the rest of the test body, so
we will never know if the test starts to pass because the bug is
fixed. Use of `pytest.xfail` is checked for, and forbidden, via
a pre-commit hook.

Instead, to handle this (hopefully rare) case, we can programmatically
mark a test as expected to fail under a combination of conditions by
applying the `pytest.mark.xfail` mark to the current test `request`.
To achieve this, the test function should take an extra parameter
named `request`, on which we call `applymarker`:

```python
@pytest.mark.parametrize("v1", [1, 2, 3])
@pytest.mark.parametrize("v2", [1, 2, 3])
def test_sum_lt_6(request, v1, v2):
    request.applymarker(
        pytest.mark.xfail(
            condition=(v1 == 3 and v2 == 3),
            reason="Add comment linking to relevant issue",
        )
    )
    assert v1 + v2 < 6
```

This way, when the bug is fixed, the test suite will fail at this
point (and we will remember to update the test).


(testing_warnings)=

### Testing code that throws warnings

Some code may be expected to throw warnings.
A common example is when a cudf API is deprecated for future removal, but many other possibilities exist as well.
The cudf testing suite [surfaces all warnings as errors](https://docs.pytest.org/en/latest/how-to/capture-warnings.html#controlling-warnings).
This includes warnings raised from non-cudf code, such as calls to pandas or pyarrow.
This setting forces developers to proactively deal with deprecations from other libraries,
as well as preventing the internal use of deprecated cudf APIs in other parts of the library.
Just as importantly, it can help catch real errors like integer overflow or division by zero.

When testing code that is expected to throw a warnings, developers should use the
[`pytest.warns`](https://docs.pytest.org/en/latest/how-to/capture-warnings.html#assertwarnings) context to catch the warning.
For parametrized tests that raise warnings under specific conditions, use the `testing._utils.expect_warning_if` decorator instead of `pytest.warns`.

```{warning}
[`warnings.catch_warnings`](https://docs.python.org/3/library/warnings.html#warnings.catch_warnings)
is a tempting alternative to `pytest.warns`.
**Do not use this context manager in tests.**
Unlike `pytest.warns`, which _requires_ that the expected warning be raised,
`warnings.catch_warnings` simply catches warnings that appear without requiring them.
The cudf testing suite should avoid such ambiguities.
```

### Testing utility functions

The `cudf.testing` subpackage provides a handful of utilities for testing the equality of objects.
The internal `cudf.testing._utils` module provides additional helper functions for use in tests.
In particular:
- `testing._utils.assert_eq` is the biggest hammer to reach for. It can be used to compare any pair of objects.
- For comparing specific objects, use `testing.testing.assert_[frame|series|index]_equal`.
- For verifying that the expected assertions are raised, use `testing._utils.assert_exceptions_equal`.


### Version testing

It is recommended to have `cudf` pytests only work on the latest supported pandas version i.e., `PANDAS_CURRENT_SUPPORTED_VERSION`. Any anticipated failures should be either `skipped` or `xfailed`.

For example:

```python
@pytest.mark.skipif(PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION, reason="bug in older version of pandas")
def test_bug_from_older_pandas_versions(...):
    ...

@pytest.mark.xfail(PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION, reason="bug in latest version of pandas")
def test_bug_in_current_and_maybe_future_versions(...):
    ...
```

If pandas makes a bugfix release and fixes this, then we'll see it in CI immediately, patch it, and bump `PANDAS_CURRENT_SUPPORTED_VERSION` which also usually happens during pandas upgrades.
