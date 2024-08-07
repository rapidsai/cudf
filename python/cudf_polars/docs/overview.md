# Getting started

You will need:

1. Rust development environment. If you use the rapids [combined
   devcontainer](https://github.com/rapidsai/devcontainers/), add
   `"./features/src/rust": {"version": "latest", "profile": "default"},` to your
   preferred configuration. Or else, use
   [rustup](https://www.rust-lang.org/tools/install)
2. A [cudf development
   environment](https://github.com/rapidsai/cudf/blob/branch-24.08/CONTRIBUTING.md#setting-up-your-build-environment).
   The combined devcontainer works, or whatever your favourite approach is.

> ![NOTE] These instructions will get simpler as we merge code in.

## Installing polars

`cudf-polars` works with polars >= 1.3, as long as the internal IR
version doesn't get a major version bump. So `pip install polars>=1.3`
should work. For development, if we're adding things to the polars
side of things, we will need to build polars from source:

```sh
git clone https://github.com/pola-rs/polars
cd polars
```

We will install build dependencies in the same environment that we created for
building cudf. Note that polars offers a `make build` command that sets up a
separate virtual environment, but we don't want to do that right now. So in the
polars clone:

```sh
# cudf environment (conda or pip) is active
pip install --upgrade uv
uv pip install --upgrade -r py-polars/requirements-dev.txt
```

> ![NOTE] plain `pip install` works fine, but `uv` is _much_ faster!

Now we have the necessary machinery to build polars
```sh
cd py-polars
# build in debug mode, best option for development/debugging
maturin develop -m Cargo.toml
```

For benchmarking purposes we should build in release mode
```sh
RUSTFLAGS='-C target-cpu=native' maturin develop -m Cargo.toml --release
```

After any update of the polars code, we need to rerun the `maturin` build
command.

## Installing the cudf polars executor

The executor for the polars logical plan lives in the cudf repo, in
`python/cudf_polars`. Build cudf as normal and then install the
`cudf_polars` package in editable mode:

```sh
cd cudf/python/cudf_polars
pip install --no-build-isolation --no-deps -e .
```

You should now be able to run the tests in the `cudf_polars` package:
```sh
pytest -v tests
```

# Executor design

The polars `LazyFrame.collect` functionality offers configuration of
the engine to use for collection through the `engine` argument. At a
low level, this provides for configuration of a "post-optimization"
callback that may be used by a third party library to replace a node
(or more, though we only replace a single node) in the optimized
logical plan with a Python callback that is to deliver the result of
evaluating the plan. This splits the execution of the plan into two
phases. First, a symbolic phase which translates to our internal
representation (IR). Second, an execution phase which executes using
our IR.

The translation phase receives the a low-level Rust `NodeTraverser`
object which delivers Python representations of the plan nodes (and
expressions) one at a time. During translation, we endeavour to raise
`NotImplementedError` for any unsupported functionality. This way, if
we can't execute something, we just don't modify the logical plan at
all: if we can translate the IR, it is assumed that evaluation will
later succeed.

The usage of the cudf-based executor is therefore selected with the
gpu engine:

```python
import polars as pl

result = q.collect(engine="gpu")
```

This should either transparently run on the GPU and deliver a polars
dataframe, or else fail (but be handled) and just run the normal CPU
execution. If `POLARS_VERBOSE` is true, then fallback is logged with a
`PerformanceWarning`.

As well as a string argument, the engine can also be specified with a
polars `GPUEngine` object. This allows passing more configuration in.
Currently, the public properties are `device`, to select the device,
and `memory_resource`, to select the RMM memory resource used for
allocations during the collection phase.

For example:
```python
import polars as pl

result = q.collect(engine=pl.GPUEngine(device=1, memory_resource=mr))
```

Uses device-1, and the given memory resource. Note that the memory
resource provided _must_ be valid for allocations on the specified
device, no checking is performed.

For debugging purposes, we can also pass undocumented keyword
arguments, at the moment, `raise_on_fail` is also supported, which
raises, rather than falling back, during translation:

```python

result = q.collect(engine=pl.GPUEngine(raise_on_fail=True))
```

This is mostly useful when writing tests, since in that case we want
any failures to propagate, rather than falling back to the CPU mode.

## IR versioning

On the polars side, the `NodeTraverser` object advertises an internal
version (via `NodeTraverser.version()` as a `(major, minor)` tuple).
`minor` version bumps are for backwards compatible changes (e.g.
exposing new nodes), whereas `major` bumps are for incompatible
changes. We can therefore attempt to detect the IR version
(independently of the polars version) and dispatch, or error
appropriately. This should be done during IR translation in
`translate.py`.

## Adding a handler for a new plan node

Plan node definitions live in `cudf_polars/dsl/ir.py`, these are
`dataclasses` that inherit from the base `IR` node. The evaluation of
a plan node is done by implementing the `evaluate` method.

To translate the plan node, add a case handler in `translate_ir` which
lives in `cudf_polars/dsl/translate.py`.

As well as child nodes that are plans, most plan nodes contain child
expressions, which should be transformed using the input to the plan as a
context. The translation of expressions is handled via
`translate_expr` in `cudf_polars/dsl/translate.py`. So that data-type
resolution is performed correctly any expression should be translated
with the correct plan node "active" in the visitor. For example, when
translating a `Join` node, the left keys (expressions) should be
translated with the left input active (and right keys with right
input). To facilitate this, use the `set_node` context manager.

## Adding a handler for a new expression node

Adding a handle for an expression node is very similar to a plan node.
Expressions are all defined in `cudf_polars/dsl/expr.py` and inherit
from `Expr`. Unlike plan nodes, these are not `dataclasses`, since it
is simpler for us to implement efficient hashing, repr, and equality if we
can write that ourselves.

Every expression consists of two types of data:
1. child data (other `Expr`s)
2. non-child data (anything other than an `Expr`)
The generic implementations of special methods in the base `Expr` base
class require that the subclasses advertise which arguments to the
constructor are non-child in a `_non_child` class slot. The
constructor should then take arguments:
```python
def __init__(self, *non_child_data: Any, *children: Expr):
```
Read the docstrings in the `Expr` class for more details.

Expressions are evaluated by implementing a `do_evaluate` method that
takes a `DataFrame` as context (this provides columns) along with an
`ExecutionContext` parameter (indicating what context we're evaluating
this expression in, currently unused) and a `mapping` from
expressions to evaluated `Column`s. This approach enables a simple form of
expression rewriting during evaluation of expressions that is used in
evaluation of, for example, groupby-aggregations. To perform the
evaluation, one should use the base class (generic) `evaluate` method
which handles the boilerplate for looking up in the substitution
`mapping`.

To simplify state tracking, all columns should be considered immutable
on construction. This matches the "functional" description coming from
the logical plan in any case, so is reasonably natural.

# Containers

Containers should be constructed as relatively lightweight objects
around their pylibcudf counterparts. We have four (in
`cudf_polars/containers/`):

1. `Scalar` (a wrapper around a pylibcudf `Scalar`)
2. `Column` (a wrapper around a pylibcudf `Column`)
3. `NamedColumn` (a `Column` with an additional name)
4. `DataFrame` (a wrapper around a pylibcudf `Table`)

The interfaces offered by these are somewhat in flux, but broadly
speaking, a `DataFrame` is just a list of `NamedColumn`s which each
hold a `Column` plus a string `name`. `NamedColumn`s are only ever
constructed via `NamedExpr`s, which are the top-level expression node
that lives inside an `IR` node. This means that the expression
evaluator never has to concern itself with column names: columns are
only ever decorated with names when constructing a `DataFrame`.

The columns keep track of metadata (for example, whether or not they
are sorted). We could imagine tracking more metadata, like minimum and
maximum, though perhaps that is better left to libcudf itself.

We offer some utility methods for transferring metadata when
constructing new dataframes and columns, both `DataFrame` and `Column`
offer a `sorted_like(like: Self)` call which copies metadata from the
template.

All methods on containers that modify in place should return `self`,
to facilitate use in a ["fluent"
style](https://en.wikipedia.org/wiki/Fluent_interface). It makes it
much easier to write iteration over objects and collect the results if
everyone always returns a value.

# Writing tests

We use `pytest`, tests live in the `tests/` subdirectory,
organisationally the top-level test files each handle one of the `IR`
nodes. The goal is that they are parametrized over all the options
each node will handle, to have reasonable coverage. Tests of
expression functionality should live in `tests/expressions/`.

To write a test an assert correctness, build a lazyframe as a query,
and then use the utility assertion function from
`cudf_polars.testing.asserts`. This runs the query using both the cudf
executor and polars CPU, and checks that they match. So:

```python
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_whatever():
    query = pl.LazyFrame(...).(...)

    assert_gpu_result_equal(query)
```

## Test coverage and asserting failure modes

Where translation of a query should fail due to the feature being
unsupported we should test this. To assert that _translation_ raises
an exception (usually `NotImplementedError`), use the utility function
`assert_ir_translation_raises`:

```python
from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_whatever():
    unsupported_query = ...
    assert_ir_translation_raises(unsupported_query, NotImplementedError)
```

This test will fail if translation does not raise.

# Debugging

If the callback execution fails during the polars `collect` call, we
obtain an error, but are not able to drop into the debugger and
inspect the stack properly: we can't cross the language barrier.

However, we can drive the translation and execution of the DSL by
hand. Given some `LazyFrame` representing a query, we can first
translate it to our intermediate representation (IR), and then execute
and convert back to polars:

```python
from cudf_polars.dsl.translate import translate_ir

q = ...

# Convert to our IR
ir = translate_ir(q._ldf.visit())

# DataFrame living on the device
result = ir.evaluate(cache={})

# Polars dataframe
host_result = result.to_polars()
```

If we get any exceptions, we can then debug as normal in Python.
