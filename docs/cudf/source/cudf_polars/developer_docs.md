(cudf-polars-developer-docs)=

# Developer Documentation

cudf-polars provides an in-memory, GPU-accelerated execution engine for Python users of the Polars Lazy API.
This page provides guidelines for the development of cudf-polars. Users should visit the main guide.

## Test Organization

cudf-polars provides many configuration options, which are bundled together into
an `engine` that executes the polars query. To ensure we have thorough test
coverage of these different engines, while avoiding duplication, cudf-polars
uses two types of tests:

1. common tests, which are run by each engine. These build some polars query and
   execute it. These tests shouldn't explicitly set an `engine`. Our test runner
   will run the test multiple times, once per engine.
2. engine-specific tests, which exercise some specific aspect of an engine. The
   tests might build a polars query or unit test some other functionality from
   `cudf-polars`. Either way, they'll also construct and use an engine with very
   specific parameters.

Our test runner in `ci/run_cudf_polars_pytests.sh` runs the test suite with
handful of engines (e.g. `in-memory`, `streaming`, `streaming-distributed`,
...). To run the appropriate tests for an engine, we ensure the on-disk layout
of the tests matches:

- common tests go under `python/cudf_polars/tests/common`
- engine specific tests go under `python/cudf_polars/test/{engine}`, where
  `{engine}` identifies one of the engines (e.g. `in-memory`).
