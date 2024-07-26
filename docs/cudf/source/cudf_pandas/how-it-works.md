# How it Works

When `cudf.pandas` is activated, `import pandas` (or any of its
submodules) imports a proxy module, rather than "regular" pandas. This
proxy module contains proxy types and proxy functions:

```python
In [1]: %load_ext cudf.pandas

In [2]: import pandas as pd

In [3]: pd
Out[3]: <module 'pandas' (ModuleAccelerator(fast=cudf, slow=pandas))>
```

Operations on proxy types/functions execute on the GPU where
possible and on the CPU otherwise, synchronizing under the hood as
needed. This applies to pandas operations both in your code and
in third-party libraries you may be using.

![cudf-pandas-execution-flow](../_static/cudf-pandas-execution-flow.png)

All `cudf.pandas` objects are a proxy to either a GPU (cuDF) or CPU
(pandas) object at any given time. Attribute lookups and method calls
are first attempted on the GPU (copying from CPU if necessary).  If
that fails, the operation is attempted on the CPU (copying from GPU if
necessary).

Additionally, `cudf.pandas` special cases chained method calls (for
example `.groupby().rolling().apply()`) that can fail at any level of
the chain and rewinds and replays the chain minimally to deliver the
correct result. Data is automatically transferred from host to device
(and vice versa) only when necessary, avoiding unnecessary device-host
transfers.

When using `cudf.pandas`, cuDF's [pandas compatibility
mode](api.options) is automatically enabled, ensuring consistency with
pandas-specific semantics like default sort ordering.


`cudf.pandas` uses a managed pool memory by default, that will also enable
prefetching <<LINK to prefetch tutorial/demo>>.

There are various memory allocators that can be used by changing the environment
variable `CUDF_PANDAS_RMM_MODE`. It supports:

1. "pool"
2. "async"
3. "managed" (default)
4. "managed_pool"
5. "cuda"
