# cudf.pandas
The use of the cuDF pandas accelerator mode (`cudf.pandas`) is explained [here](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/). The purpose of this document is to explain how the fast-slow proxy mechanism works and document the internal environment variables used to debug `cudf.pandas`.

## fast-slow proxy mechanism
`cudf.pandas` works by wrapping each Pandas type (and its corresponding cuDF type) in new proxy types (aka fast-slow proxy types). Because the proxy types wrap both the fast and slow implementations of the original type, we can ensure that computations are first done on the fast version of the proxy type, and if that fails, the slow version of the proxy type.

### Types
#### Wrapped Types and Proxy Types
The "wrapped" types/classes are the Pandas and cuDF specific types that have been wrapped into proxy types. Wrapped objects and proxy objects are instances of wrapped types and proxy types, respectively. In the snippet below `s1` and `s2` are wrapped objects and `s3` is a fast-slow proxy object. Also note that the module `xpd` is a wrapped module and contains cuDF and Pandas modules as attributes.
  ```python
  import cudf.pandas
  cudf.pandas.install()
  import pandas as xpd

  cudf = xpd._fsproxy_fast
  pd = xpd._fsproxy_slow

  s1 = cudf.Series([1,2])
  s2 = pd.Series([1,2])
  s3 = xpd.Series([1,2])
  ```

#### The Different Kinds of Proxy Types
In `cudf.pandas`, proxy types come in multiple kinds: fast-slow proxy types, callable types, and fast-slow attribute types.

Fast-slow proxy types come in two flavors: final types and intermediate types. Final types are types for which known operations exist for converting an object of "fast" type to "slow" and vice-versa. For example, `cudf.DataFrame` can be converted to Pandas using the method `to_pandas` and `pd.DataFrame` can be converted to cuDF using the function `cudf.from_pandas`.

Intermediate types are the types of the results of operations invoked on final types. For example, `DataFrameGroupBy` is a type that will be created during a groupby operation.

Callable types are ...

Fast-Slow attribute types ...

#### Creating New Proxy Types
`_FinalProxy` and `_IntermediateProxy` types are created using the functions `make_final_proxy_type` and `make_intermediate_proxy` type, respectively. Creating a new final type looks like this.

```python
DataFrame = make_final_proxy_type(
    "DataFrame",
    cudf.DataFrame,
    pd.DataFrame,
    fast_to_slow=lambda fast: fast.to_pandas(),
    slow_to_fast=cudf.from_pandas,
)
```

### The Fallback Mechanism
The `_fast_slow_function_call` is the all-important function that we use to call operations the fast way (using cuDF) and if that fails, the slow way (using Pandas). The is also known as the fallback mechanism. The function looks like this:
```python
def _fast_slow_function_call(func: Callable, *args, **kwargs):
    try:
        ...
        fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
        result = func(*fast_args, **fast_kwargs)
        ...
    except Exception:
        ...
        slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
        result = func(*slow_args, **slow_kwargs)
        ...
```
As you can see the function attempts to call `func` the fast way using cuDF and if an any `Exception` occurs, it calls the function using Pandas. In essence, this `try-except` is what allows `cudf.pandas` to support 100% of the Pandas API.

### Using Metaclasses
`cudf.pandas` uses a [metalass](https://docs.python.org/3/glossary.html#term-metaclass) called (`_FastSlowProxyMeta`) to dynamically find class attributes and classmethods of fast-slow proxy types. For example, in the snippet below, the `xpd.Series` type is an instance `_FastSlowProxyMeta`. Therefore we can access the property `_fsproxy_fast` defined in the metaclass.
```python
import cudf.pandas
cudf.pandas.install()
import pandas as xpd

print(xpd.Series._fsproxy_fast) # output is cudf.core.series.Series
```

### Caching

### Pickling and Unpickling

## debugging `cudf.pandas`
Several environment variables are available for debugging purposes.

Setting the environment variable `CUDF_PANDAS_DEBUGGING` produces a warning when the results from cuDF and Pandas differ from one another. For example, the snippet below produces the warning below.
```python
import cudf.pandas
cudf.pandas.install()
import pandas as pd
import numpy as np

setattr(pd.Series.mean, "_fsproxy_slow", lambda self, *args, **kwargs: np.float64(1))
s = pd.Series([1,2,3])
s.mean()
```
```
UserWarning: The results from cudf and pandas were different. The exception was
Arrays are not almost equal to 7 decimals
 ACTUAL: 1.0
 DESIRED: 2.0.
```
