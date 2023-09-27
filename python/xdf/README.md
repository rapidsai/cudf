# xdf: Unified GPU/CPU DataFrame library

`xdf` is a DataFrame library that uses the GPU (via
[cuDF](https://github.com/rapidsai/cudf)) when it can, and falls back
to using the CPU (via [pandas](https://pandas.pydata.org)) for
operations not supported on the GPU or when a GPU is not available.

## Usage

`xdf` works by providing importable modules that offer the same API
as pandas. You can access them in one of two ways. Either by enabling
"transparent" mode (which redirects all imports of pandas modules to
their xdf equivalents), or by explicit import of `xdf.pandas` (and
submodules) in place of imports of `pandas`. Our recommendation is to
use transparent mode, since that requires minimal code changes, and
works most straightforwardly with third-party libraries that also use
pandas datatypes.

### Activating xdf in transparent mode

This mode works by intercepting all imports of pandas modules and
pointing them at the equivalent xdf module. For this to work, pandas
cannot already have been imported, and a `RuntimeError` is raised if
this situation is detected. It can be used with no, or only minimal
source file changes.

To replace use of pandas with xdf without source file changes, run
your python script with:
```shell
$ python -m xdf.autoload script.py
```

The `xdf.autoload` module also offers a programmatic interface for
installation:
```python
import xdf.autoload
xdf.autoload.install()
```
It is also usable as a [pytest](https://pytest.org) plugin to replace
usage of `pandas` in a test suite with `xdf`:
```shell
$ pytest -p xdf.autoload tests/
```

Note that this usage mode is a global setting, and you cannot control
whether xdf or pandas objects will be created on a case-by-case basis.

### Importing xdf as a library

If you do not wish to use transparent mode, you can instead import the
xdf modules manually as a replacement for pandas imports. In this mode
the `xdf.pandas` module hierarchy is designed to mimic the pandas API,
so that you can replace `import pandas as pd` with `import xdf.pandas
as pd`. For example:
```python
import numpy as np
import xdf.pandas as xpd
df = xpd.DataFrame({'a': np.random.randint(0, 101, 1_000_000), 'b': np.random.rand(1_000_000)})
result = df.groupby('a').max()  # fast: uses the GPU
func = lambda group: {'min': group['b'].min(), 'max': group['b'].max()}
result = df.groupby('a').apply(func)  # slow: uses the CPU as `func` cannot execute on the GPU
```

Note that activating xdf in transparent mode and using the
`xdf.pandas` module hierarchy simultaneously is not possible, and a
`RuntimeError` will be raised if this usage mode is detected.

### Disabling xdf when running an individual script

If `XDF_FALLBACK_MODE` is in the calling environment, then the above
mechanisms for importing xdf work as normal, except that the fast
library is not used for acceleration, and xdf only uses pandas.
For example:
```shell
# import pandas uses only pandas
$ XDF_FALLBACK_MODE=1 python -m xdf.autoload some_script.py
# import pandas uses xdf (and hence cudf)
$ python -m xdf.autoload some_script.py
```
The same is also true if you use `import xdf.pandas as pandas`.

## How xdf works

`xdf` _extends_ cuDF by providing _all_ of the Pandas API.
When a function, or arguments to a function are not supported by cuDF,
`xdf` will transparently:

1. copy the input data from the GPU to the CPU
2. perform the operation on CPU using Pandas
3. copy the result back from CPU to GPU

`xdf` provides a profiler that can report whether operations
used the CPU or the GPU:

```python
import xdf
import xdf.pandas as xpd

with xdf.Profiler() as p:
    df = xpd.DataFrame({"a": [0, 1, 2], "b": "a"})
    df.max(skipna=True)
    for i in range(0, 2):
        df.min(axis=0)
        axis = 1
    df.groupby("a").max()
    df.groupby("a").filter(lambda group: len(group) > 1)
    df.apply(lambda s: s.tolist(), axis=1)
    df.groupby("a").max().apply(lambda s: s.tolist(), axis=1)

p.print_stats()
                                                 Stats
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Line no. ┃ Line                                                          ┃ GPU TIME(s) ┃ CPU TIME(s) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 4        │     df = xpd.DataFrame({"a": [0, 1, 2], "b": "a"})            │ 0.911555052 │             │
│          │                                                               │             │             │
│ 5        │     df.max(skipna=True)                                       │ 0.000163317 │ 0.010112762 │
│          │                                                               │             │             │
│ 6        │     for i in range(0, 2):                                     │             │             │
│          │                                                               │             │             │
│ 7        │         df.min(axis=0)                                        │ 0.005092621 │ 0.015736103 │
│          │                                                               │             │             │
│ 8        │         axis = 1                                              │             │             │
│          │                                                               │             │             │
│ 9        │     df.groupby("a").max()                                     │ 0.006950378 │             │
│          │                                                               │             │             │
│ 10       │     df.groupby("a").filter(lambda group: len(group) > 1)      │ 0.000334024 │ 0.004799128 │
│          │                                                               │             │             │
│ 11       │     df.apply(lambda s: s.tolist(), axis=1)                    │ 0.002256632 │ 0.001369476 │
│          │                                                               │             │             │
│ 12       │     df.groupby("a").max().apply(lambda s: s.tolist(), axis=1) │ 0.006145239 │ 0.001305819 │
│          │                                                               │             │             │
└──────────┴───────────────────────────────────────────────────────────────┴─────────────┴─────────────┘
```

Notice that when falling back to the CPU,
a small amount of time is first spent attempting to use the GPU.

## Installation

This project is under development and pre-built pip/conda packages are not yet
available.  The easiest way to install `xdf` is from this GitHub
repository:

```
pip install git+https://github.com/rapidsai/xdf.git
```

## Developing `xdf`

### Setting up xdf for development

#### Install `cudf`

While it is not a strict dependency, `cudf` will be used by `xdf` if present in the environment. Otherwise, Pandas will be used for all operations.
Thus, you may want to install `xdf` into your existing cuDF
development environment. Alternately, if you just want to hack on `xdf`
without touching any of the cuDF source, you can begin by
[installing cuDF](https://docs.rapids.ai/install#selector).

#### Install `xdf`

To set up `xdf` for development locally, first clone the repository to
get the source:

```
git clone https://github.com/rapidsai/xdf.git
```

Install in editable mode. The following command will install test
dependencies as well:

```
cd xdf
pip install -e .[test]
```

Changes to the source are now reflected in the install.

### Testing

#### Unit tests

To run the unit tests:

```
pytest tests/
```

#### Running Pandas unit test suite with `xdf`

In addition to its own unit tests, `xdf` is tested against the
Pandas test suite. At the time of writing, not all Pandas unit tests
are passed by `xdf` and this is an active area of development.

The `run-pandas-tests.sh` script is used to run Pandas unit tests
using `xdf`. Arguments to this script are forwarded to `pytest`. An
example invocation of `run-pandas-tests.sh` is:

```
bash scripts/run-pandas-tests.sh --rewrite-imports -n auto -v --tb=line --skip-slow --report-log=pandas-full.json
```

The above command replaces all imports of `pandas` with imports of
`xdf.pandas` and then runs the entire Pandas test suite (skipping tests with
the pytest mark `slow`) writing a report to the file
`pandas-testing/pandas-full.json`.  This report can be summarized
using the script `summarize-test-results.py` as follows:

```
python scripts/summarize-test-results.py pandas-testing/pandas-full.json

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Test module                    ┃ Total tests ┃ Passed tests ┃ Failed tests ┃ Errored tests ┃ Skipped tests ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ test_errors.py                 │ 32          │ 32           │ 0            │ 0             │ 0             │
│ arrays/test_ndarray_backed.py  │ 5           │ 5            │ 0            │ 0             │ 0             │
│ arrays/test_period.py          │ 19          │ 19           │ 0            │ 0             │ 0             │

...
```

Finally, it is useful to be able to see the most common types of failures
for a particular test module or set of test modules. The script
`analyze-test-failures.py` can be used for this purpose:

```
python scripts/analyze-test-failures.py pandas-testing/pandas-full.json "arithmetic/test_datetime64.py"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Failure message                                                                   ┃ Number of occurences ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ AssertionError: [5000000000 5000000000]                                           │ 280                  │
│ AssertionError: [              0  86400000000000 172800000000000 259200000000000  │ 160                  │
│ AttributeError: 'numpy.datetime64' object has no attribute 'to_pydatetime'        │ 160                  │
│ AssertionError: Attributes of Series are different                                │ 128                  │
│ AssertionError: Regex pattern did not match.                                      │ 46                   │
...
```

### Code formatting

To format your code before submitting a PR, use `pre-commit`:

```
pre-commit --run all-files
```

You can optionally set up `pre-commit` to run before every git commit:

```
pre-commit install
```


### Design notes

See [design.md](design.md) for some notes about the design of `xdf`
and how it works.
