# How it works
When using cuDF’s pandas accelerator mode, operations execute on the GPU where
possible and on the CPU otherwise, synchronizing under the hood as needed.

All `cudf.pandas` objects are a proxy to either a GPU (cuDF) or CPU (Pandas)
object at any given time. Attribute lookups and method calls are first
attempted on the GPU object. If that fails, they’re attempted on the CPU
object. Additionally, `cudf.pandas` special cases chained method calls (for
example `groupby-apply`) that can fail at any level of the chain and rewinds
and replays the chain minimally to deliver a correct result.and/or minimizes
roundtrips between cuDF and Pandas objects. cuDF operations are run in pandas
compatibility mode, ensuring consistency with pandas-specific semantics like
default sort and join ordering.

## Understanding Performance

When using cuDF’s pandas accelerator mode, operations execute on the GPU where
possible and on the CPU otherwise, synchronizing under the hood as needed.

As a result, it’s possible a workflow will run some operations on the GPU and
some on the CPU, depending on the specific details.

cuDF Pandas Accelerator Mode provides a profiling utility to help understand
performance and provide visibility into which operations used the GPU or the
CPU.


```python
%load_ext cudf.pandas
import pandas as pd
```

```python
%%cudf.pandas.profile

df = pd.DataFrame({'a': [0, 1, 2], 'b': "a"})
df.max(skipna=True)

axis = 0
for i in range(0, 2):
	df.min(axis=axis)
	axis = 1

out = df.groupby('a').filter(
	lambda group: len(group) > 1
)
```

![cudf.pandas profile](../_static/cudf.pandas-profile.png)


## How is this different from other DataFrame-like libraries?


1. Designed for an exact and accelerated Pandas experience
2. Designed for when Pandas is too slow but data sizes are reasonably computed on
with single machine e.g. not big data
3. It accelerates and integrates with 3rd party libraries without asking
maintainers to make any changes
4. We explicitly test against the Pandas test suite and are working towards 100%
test suite coverage


## How We Ensure Consistency with Pandas

Pandas unit testing
Integration testing

Learn more in the #pandas_coverage FAQ

