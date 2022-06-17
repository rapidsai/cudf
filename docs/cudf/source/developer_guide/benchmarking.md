# Benchmarking cuDF

The goal of the benchmarks in this repository is to measure the performance of various cuDF APIs.
Benchmarks in cuDF are written using the
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) plugin to the
[`pytest`](https://docs.pytest.org/en/latest/) Python testing framework.
Using `pytest-benchmark` provides a seamless experience for developers familiar with `pytest`.
We include benchmarks of both public APIs and internal functions
The former give us a macro view of our performance, especially vis a vis pandas.
The latter help us quantify and minimize the overhead of our Python bindings.

```{note}
Minimizing memory footprint can be just as important as minimizing run time.
Our benchmarks do not currently account for memory usage at all.
However, we may reconsider this in the future.
```

## Benchmark organization

At the top level benchmarks are divided into `internal` and `API` directories.
API benchmarks are for public features that we expect users to consume.
Internal benchmarks capture the performance of cuDF internals that have no stability guarantees.

Within each directory, benchmarks are organized based on the type of function.
Functions in cuDF generally fall into two groups:

1. Methods of classes like `DataFrame` or `Series`.
2. Free functions operating on the above classes like `cudf.merge`.

The former should be organized into files based on the class.
API benchmark files should be named `bench_$class.py`.
For example, benchmarks of `DataFrame.eval` belong in `API/bench_dataframe.py`.
For benchmarks of internal classes (or benchmarks of internal methods of public classes),
the class files should be suffixed with `_internal` to avoid naming conflicts.
Even if they are in separate directories, identically named files can cause issues for `pytest`.
Benchmarks of `DataFrame._apply_boolean_mask`, for instance, belong in `internal/bench_dataframe_internal.py`.

To ensure that benchmarks cover the widest range of classes possible,
benchmarks should be organized into files corresponding to the highest class in the hierarchy that they support.
For instance, all classes support the `take` method, so those benchmarks belong in `API/bench_frame_or_index.py`.
Some APIs exist for both Index and IndexedFrame classes, but with slightly different APIs.
In such cases, benchmarks should be written using to a minimal common API,
_unless_ developers expect certain arguments to trigger code paths with very different performance characteristics.

Free functions have more flexibility.
Broadly speaking, they should be grouped into benchmark files containing similar functionality.
We may evolve a more concrete set of guidelines on exactly what that should look like over time,
but for now those groupings are left to the discretion of developers.


## Standard object fixtures

Benchmarks of methods are typically straightforward to write.
Using `pytest-benchmark`'s benchmark fixture, they are typically as simple as `benchmark(obj.method)`.
One of the most common difficulties, however, is deciding exactly what `obj` should look like.
Developers often want to benchmark objects of different sizes, or multiple classes.
Conversely, we would like to avoid proliferating nearly identical functions for object creation.

`pytest` is designed to address this problem using fixtures.
For our purposes, however, we also need to be able to run some benchmarks for _multiple_ fixtures.
For instance, benchmarks in `API/bench_indexedframe.py` must be run for both `DataFrame` and `Series`.
To address this requirement, we make use of [`pytest_cases`](https://smarie.github.io/python-pytest-cases/).
Specifically we use `pytest_cases.fixture_union` to create fixtures from other fixtures.
Tests parametrized by fixture unions are run once for each member of the union.

This feature is critical to the design of our benchmarks.
It allows developers to write a single benchmark `def bench_foo(indexedframe)`,
and then have that benchmark automatically run many times with different parameters.
In particular, our standard fixtures cover the following:

- Class: Objects of a specific class, e.g. `DataFrame`.
- Nullability: Objects with and without null entries.
- Dtype: Objects of a specific dtype.
- Size: Objects with a certain number of rows or columns.

Each benchmark should use the appropriate fixture to span the desired subset of this parameter space.
Where absolutely necessary, developers may define custom fixtures for a subset of benchmarks.
However, new benchmarks should be added with care.
The default benchmark fixtures are carefully designed to automatically support most case.
They will also work for the special cases like comparing to pandas or running tests, as described below.
Most importantly, they ensure reasonable benchmark coverage without ballooning the parameter space unnecessarily.

### The `accepts_cudf_fixture` decorator

The standard fixtures described above are convenient for generating benchmarks.
However, the long names required to disambiguate all the parameters are cumbersome when writing tests.
Moreover, having this information embedded in the name means that in order to change the parameters used,
the entire benchmark needs to have the fixture name replaced.

To avoid this problem, our benchmarks provide the `accepts_cudf_fixture` decorator.
This decorator allows developers to write benchmarks using a simple object name, such as `"df"`,
and then request the desired parameters using the decorator.
The decorator takes care of remapping the real fixture onto the alias used by the developer.
For example, a benchmark in `bench_dataframe.py` might look like this:

```python
@accepts_cudf_fixture(cls="dataframe", dtype="int", nulls=False, cols=6, name="df")
def bench_foo(benchmark, df):
    benchmark(df.foo)
```

In general, the type of objects being benchmarked should match the name of the benchmark file.
In the above example, for instance, we are benchmarking DataFrame objects in `bench_dataframe.py`.
We do so by specifying `cls="dataframe"`, but remap that to the name `"df"` for convenience.


## Parametrization vs fixtures

Another important feature of `pytest_cases` is how it allows us to handle parametrization.
Specifically, it provides some syntactic sugar around

```python
@pytest.mark.parametrize(
    "num", [1, 2, 3]
)
def bench_foo(benchmark, num):
    benchmark(num * 2)
```

for when the parameters are nontrivial and require complex initialization.
This is common for benchmarks of functions accepting cuDF objects, such as `cudf.concat`.
With `pytest_cases`, the different cases are instead placed into separate functions and automatically made available.

```python
# bench_foo_cases.py
def case_1():
    return 1

def case_2():
    return 2

def case_3():
    return 3

# bench_foo.py
@pytest_cases.parametrize_with_cases(num)
def bench_foo(benchmark, num):
    benchmark(num * 2)
```

This approach forces developers to put complex initialization into named, documented functions.
That becomes especially valuable when benchmarking APIs whose performance can vary drastically based on parameters.
Cases also offer one other major benefit: they are lazily evaluated.
Initializing complex objects inside a `pytest.mark.parametrize` can dramatically slow down test collection,
or even lead to out of memory issues if too many complex cases are collected.
Since case functions are lazily evaluated, the associated objects are only created on an as-needed basis.

The observant reader may recognize that cases seem quite familiar to fixtures.
In fact, `pytest_cases` generalizes the `pytest` to allow things like parametrizing with fixtures.
For our purposes, however, it is important to keep the two distinct.
From a conceptual standpoint, fixtures are homogeneous and generic while cases are heterogeneous and specific.
Fixtures should be created sparingly and used broadly.
Cases can be precisely tailored for a single test.

## Comparing to pandas

An important aspect of benchmarking cuDF is comparing it to pandas.
We often want to generate quantitative comparisons, so we need to make that as easy as possible.
Our benchmarks support this by setting the environment variable `CUDF_BENCHMARKS_USE_PANDAS`.
When this variable is detected, all benchmarks will automatically be run using pandas instead of cuDF.
Therefore, comparisons can easily be generated by simply running the benchmarks twice,
once with the variable set and once without.

## Continuous integration

Benchmarks need to be kept up to date with API changes in cuDF.
However, we cannot simply benchmarks in CI.
Doing so would consume too many resources, _and_ it would significantly slow down the development cycle

To balance these issues, our benchmarks also support running in "testing" mode.
To do so, developers can set the `CUDF_BENCHMARKS_TEST_ONLY` environment variable.
When benchmarks are run with this variable, all data sizes are set to a minimum and the number of sizes are reduced.

```{warning}
The `CUDF_BENCHMARKS_TEST_ONLY` fixture relies on the configuration values defined in `common/config.py`.
All the standard fixtures automatically respect the `NUM_ROWS` and `NUM_COLS` variables defined there.
If developers define custom fixtures or cases, they are responsible for importing and using those variables.
```

## Profiling

Although not strictly part of our benchmarking suite, profiling is a common need so we provide some guidelines here.
There are two easy ways to profile benchmarks:
1. The [`pytest-profiling`](https://github.com/man-group/pytest-plugins/tree/master/pytest-profiling) plugin.
2. The [`py-spy`](https://github.com/benfred/py-spy) package.

Using the former is as simple as adding the `--profile` (or `--profile-svg`) arguments to your `pytest` invocation.
The latter requires instead invoking pytest from py-spy, like so:
```
py-spy record -- pytest bench_foo.py
```
Depending on exactly what information you need, your mileage may vary with each one.
Developers should try both and see what works for their workflows.
