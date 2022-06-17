# Benchmarking cuDF

The goal of the benchmarks in this repository is to measure the performance of various cuDF APIs.
Benchmarks in cuDF are written using the
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) plugin to the
[`pytest`](https://docs.pytest.org/en/latest/) Python testing framework.
Using `pytest-benchmark` provides a seamless experience for developers familiar with `pytest`.
We include benchmarks of both public APIs and internal functions.
The former give us a macro view of our performance, especially vis a vis pandas.
The latter help us quantify and minimize the overhead of our Python bindings.

```{note}
Our current benchmarks focus entirely on measuring run time.
However, minimizing memory footprint can be just as important for some cases.
In the future, we may update our benchmarks to also include memory usage measurements.
```

## Benchmark organization

At the top level benchmarks are divided into `internal` and `API` directories.
API benchmarks are for public features that we expect users to consume.
Internal benchmarks capture the performance of cuDF internals that have no stability guarantees.

Within each directory, benchmarks are organized based on the type of function.
Functions in cuDF generally fall into two groups:

1. Methods of classes like `DataFrame` or `Series`.
2. Free functions operating on the above classes like `cudf.merge`.

The former should be organized into files named `bench_class.py`.
For example, benchmarks of `DataFrame.eval` belong in `API/bench_dataframe.py`.
Benchmarks should be written at the highest level of generality possible with respect to the class hierarchy.
For instance, all classes support the `take` method, so those benchmarks belong in `API/bench_frame_or_index.py`.
If a method has a slightly different API for different classes, benchmarks should use a minimal common API,
_unless_ developers expect certain arguments to trigger code paths with very different performance characteristics.

```{note}
`pytest` does not support having two benchmark files with the same name, even if they are in separate directories.
Therefore, benchmarks of internal methods of _public_ classes go in files suffixed with `_internal`.
Benchmarks of `DataFrame._apply_boolean_mask`, for instance, belong in `internal/bench_dataframe_internal.py`.
```

Free functions have more flexibility.
Broadly speaking, they should be grouped into benchmark files containing similar functionality.
For example, I/O benchmarks can all live in `bench_io.py`.
For now those groupings are left to the discretion of developers.

## Standard fixtures

The most difficult part of writing a benchmark is often deciding what class of objects that benchmark should run for.
Developers often need to run the same benchmarks with _multiple_ `pytest` fixtures rather than just one.
For instance, benchmarks in `API/bench_indexedframe.py` must be run for both `DataFrame` and `Series`.
To address this requirement, we make use of [`pytest_cases`](https://smarie.github.io/python-pytest-cases/).
Specifically we use `pytest_cases.fixture_union` to create fixture unions.
Tests parametrized by fixture unions are run once for each member of the union.

This feature is critical to the design of our benchmarks.
It allows developers to write a single benchmark `def bench_foo(indexedframe)`,
and then have that benchmark automatically run many times with different parameters.
In particular, our standard fixtures cover the following:

- Class: Objects of a specific class, e.g. `DataFrame`.
- Nullability: Objects with and without null entries.
- Dtype: Objects of a specific dtype.
- Size: Objects with a certain number of rows or columns.

Fixture names identify these parameters unambiguously like so:
`{classname}_dtype_{dtype}[_nulls_{true|false}][[_cols_{num_cols}]_rows_{num_rows}]`.
If a fixture name does not contain a particular component, it represents a union of all values of that component.
For example, a benchmark
```
def bench_foo(benchmark, dataframe_dtype_int_rows_100):
    benchmark(df.foo)
```
will run for both nullable and non-nullable 100 row integer `DataFrame`s of with different numbers of columns.

These fixtures should support most use cases.
Developers may define custom fixtures if necessary, but this should be done with care.
The default fixtures provide reasonable benchmark coverage without excessive resource usage.
More bespoke fixtures, if necessary, should be constructed with the same constraints in mind.
Furthermore, the default fixtures are designed to work when
[comparing to pandas](pandascompare) or [running tests](testing).
New fixtures must also account for these use cases.

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

This code benchmarks `DataFrame` objects (`cls="dataframe"`) but remaps them to the name `"df"` for convenience.


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

(pandascompare)=

## Comparing to pandas

An important aspect of benchmarking cuDF is comparing it to pandas.
We often want to generate quantitative comparisons, so we need to make that as easy as possible.
Our benchmarks support this by setting the environment variable `CUDF_BENCHMARKS_USE_PANDAS`.
When this variable is detected, all benchmarks will automatically be run using pandas instead of cuDF.
Therefore, comparisons can easily be generated by simply running the benchmarks twice,
once with the variable set and once without.

```{warning}
`CUDF_BENCHMARKS_USE_PANDAS` relies on benchmarks importing `cudf` and `cupy` from `common/config.py`.
That allows configuration of these modules based on the environment variable.
When developers add these imports to a file, they must import from `config.py` rather than importing directly.
```

(testing)=

## Testing

Benchmarks need to be kept up to date with API changes in cuDF.
However, we cannot simply run benchmarks in CI.
Doing so would consume too many resources, and it would significantly slow down the development cycle

To balance these issues, our benchmarks also support running in "testing" mode.
To do so, developers can set the `CUDF_BENCHMARKS_TEST_ONLY` environment variable.
When benchmarks are run with this variable, all data sizes are set to a minimum and the number of sizes are reduced.
Our CI testing takes advantage of this to ensure that benchmarks remain valid code.

```{warning}
`CUDF_BENCHMARKS_TEST_ONLY` relies on the configuration values defined in `common/config.py`.
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
