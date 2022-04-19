---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Overview of User Defined Functions with cuDF

```{code-cell} ipython3
import cudf
from cudf.datasets import randomdata
import numpy as np
```

Like many tabular data processing APIs, cuDF provides a range of composable, DataFrame style operators. While out of the box functions are flexible and useful, it is sometimes necessary to write custom code, or user-defined functions (UDFs), that can be applied to rows, columns, and other groupings of the cells making up the DataFrame.

In conjunction with the broader GPU PyData ecosystem, cuDF provides interfaces to run UDFs on a variety of data structures. Currently, we can only execute UDFs on numeric, boolean, datetime, and timedelta typed data (support for strings is being planned). This guide covers writing and executing UDFs on the following data structures:

- Series
- DataFrame
- Rolling Windows Series
- Groupby DataFrames
- CuPy NDArrays
- Numba DeviceNDArrays

It also demonstrates cuDF's default null handling behavior, and how to write UDFs that can interact with null values.

+++

## Series UDFs

You can execute UDFs on Series in two ways:

- Writing a standard python function and using `cudf.Series.apply`
- Writing a Numba kernel and using Numba's `forall` syntax

Using `apply` or is simpler, but writing a Numba kernel offers the flexibility to build more complex functions (we'll be writing only simple kernels in this guide).

+++

#  `cudf.Series.apply`

+++

cuDF provides a similar API to `pandas.Series.apply` for applying scalar UDFs to series objects. Here is a very basic example.

```{code-cell} ipython3
# Create a cuDF series
sr = cudf.Series([1, 2, 3])
```

UDFs destined for `cudf.Series.apply` might look something like this:

```{code-cell} ipython3
# define a scalar function
def f(x):
    return x + 1
```

`cudf.Series.apply` is called like `pd.Series.apply` and returns a new `Series` object:

```{code-cell} ipython3
sr.apply(f)
```

### Functions with Additional Scalar Arguments

+++

In addition, `cudf.Series.apply` supports `args=` just like pandas, allowing you to write UDFs that accept an arbitrary number of scalar arguments. Here is an example of such a function and it's API call in both pandas and cuDF:

```{code-cell} ipython3
def g(x, const):
    return x + const
```

```{code-cell} ipython3
# cuDF apply
sr.apply(g, args=(42,))
```

As a final note, `**kwargs` is not yet supported.

+++

### Nullable Data

+++

The null value `NA` an propagates through unary and binary operations. Thus, `NA + 1`, `abs(NA)`, and `NA == NA` all return `NA`. To make this concrete, let's look at the same example from above, this time using nullable data:

```{code-cell} ipython3
# Create a cuDF series with nulls
sr = cudf.Series([1, cudf.NA, 3])
sr
```

```{code-cell} ipython3
# redefine the same function from above
def f(x):
    return x + 1
```

```{code-cell} ipython3
# cuDF result
sr.apply(f)
```

Often however you want explicit null handling behavior inside the function. cuDF exposes this capability the same way as pandas, by interacting directly with the `NA` singleton object. Here's an example of a function with explicit null handling:

```{code-cell} ipython3
def f_null_sensitive(x):
    # do something if the input is null
    if x is cudf.NA:
        return 42
    else:
        return x + 1
```

```{code-cell} ipython3
# cuDF result
sr.apply(f_null_sensitive)
```

In addition, `cudf.NA` can be returned from a function directly or conditionally. This capability should allow you to implement custom null handling in a wide variety of cases.

+++

### Lower level control with custom `numba` kernels

+++

In addition to the Series.apply() method for performing custom operations, you can also pass Series objects directly into [CUDA kernels written with Numba](https://numba.pydata.org/numba-doc/latest/cuda/kernels.html).
Note that this section requires basic CUDA knowledge. Refer to [numba's CUDA documentation](https://numba.pydata.org/numba-doc/latest/cuda/index.html) for details.

The easiest way to write a Numba kernel is to use `cuda.grid(1)` to manage thread indices, and then leverage Numba's `forall` method to configure the kernel for us. Below, define a basic multiplication kernel as an example and use `@cuda.jit` to compile it.

```{code-cell} ipython3
df = randomdata(nrows=5, dtypes={'a':int, 'b':int, 'c':int}, seed=12)
```

```{code-cell} ipython3
from numba import cuda

@cuda.jit
def multiply(in_col, out_col, multiplier):
    i = cuda.grid(1)
    if i < in_col.size: # boundary guard
        out_col[i] = in_col[i] * multiplier
```

This kernel will take an input array, multiply it by a configurable value (supplied at runtime), and store the result in an output array. Notice that we wrapped our logic in an `if` statement. Because we can launch more threads than the size of our array, we need to make sure that we don't use threads with an index that would be out of bounds. Leaving this out can result in undefined behavior.

To execute our kernel, must pre-allocate an output array and leverage the `forall` method mentioned above. First, we create a Series of all `0.0` in our DataFrame, since we want `float64` output. Next, we run the kernel with `forall`. `forall` requires us to specify our desired number of tasks, so we'll supply in the length of our Series (which we store in `size`). The [__cuda_array_interface__](https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html) is what allows us to directly call our Numba kernel on our Series.

```{code-cell} ipython3
size = len(df['a'])
df['e'] = 0.0
multiply.forall(size)(df['a'], df['e'], 10.0)
```

After calling our kernel, our DataFrame is now populated with the result.

```{code-cell} ipython3
df.head()
```

This API allows a you to theoretically write arbitrary kernel logic, potentially accessing and using elements of the series at arbitrary indices and use them on cuDF data structures. Advanced developers with some CUDA experience can often use this capability to implement iterative transformations, or spot treat problem areas of a data pipeline with a custom kernel that does the same job faster.

+++

## DataFrame UDFs

Like `cudf.Series`, there are multiple ways of using UDFs on dataframes, which essentially amount to UDFs that expect multiple columns as input:

- `cudf.DataFrame.apply`, which functions like `pd.DataFrame.apply` and expects a row udf
- `cudf.DataFrame.apply_rows`, which is a thin wrapper around numba and expects a numba kernel
- `cudf.DataFrame.apply_chunks`, which is similar to `cudf.DataFrame.apply_rows` but offers lower level control.

+++

# `cudf.DataFrame.apply`

+++

`cudf.DataFrame.apply` is the main entrypoint for UDFs that expect multiple columns as input and produce a single output column. Functions intended to be consumed by this API are written in terms of a "row" argument. The "row" is considered to be like a dictionary and contains all of the column values at a certain `iloc` in a `DataFrame`. The function can access these values by key within the function, the keys being the column names corresponding to the desired value. Below is an example function that would be used to add column `A` and column `B` together inside a UDF.

```{code-cell} ipython3
def f(row):
    return row['A'] + row['B']
```

Let's create some very basic toy data containing at least one null.

```{code-cell} ipython3
df = cudf.DataFrame({
    'A': [1,2,3],
    'B': [4,cudf.NA,6]
})
df
```

Finally call the function as you would in pandas - by using a lambda function to map the UDF onto "rows" of the DataFrame: 

```{code-cell} ipython3
df.apply(f, axis=1)
```

The same function should produce the same result as pandas:

```{code-cell} ipython3
df.to_pandas(nullable=True).apply(f, axis=1)
```

Notice that Pandas returns `object` dtype - see notes on this in the caveats section.

+++

Like `cudf.Series.apply`, these functions support generalized null handling. Here's a function that conditionally returns a different value if a certain input is null:

```{code-cell} ipython3
def f(row):
    x = row['a']
    if x is cudf.NA:
        return 0
    else:
        return x + 1

df = cudf.DataFrame({'a': [1, cudf.NA, 3]})
df
```

```{code-cell} ipython3
df.apply(f, axis=1)
```

`cudf.NA` can also be directly returned from a function resulting in data that has the the correct nulls in the end, just as if it were run in Pandas. For the following data, the last row fulfills the condition that `1 + 3 > 3` and returns `NA` for that row:

```{code-cell} ipython3
def f(row):
    x = row['a']
    y = row['b']
    if x + y > 3:
        return cudf.NA
    else:
        return x + y

df = cudf.DataFrame({
    'a': [1, 2, 3], 
    'b': [2, 1, 1]
})
df
```

```{code-cell} ipython3
df.apply(f, axis=1)
```

Mixed types are allowed, but will return the common type, rather than object as in Pandas. Here's a null aware op between an int and a float column:

```{code-cell} ipython3
def f(row):
     return row['a'] + row['b']

df = cudf.DataFrame({
    'a': [1, 2, 3], 
    'b': [0.5, cudf.NA, 3.14]
})
df
```

```{code-cell} ipython3
df.apply(f, axis=1)
```

Functions may also return scalar values, however the result will be promoted to a safe type regardless of the data. This means even if you have a function like:

```python
def f(x):
    if x > 1000:
        return 1.5
    else:
        return 2
```
And your data is:
```python
[1,2,3,4,5]
```
You will get floats in the final data even though a float is never returned. This is because Numba ultimately needs to produce one function that can handle any data, which means if there's any possibility a float could result, you must always assume it will happen. Here's an example of a function that returns a scalar in some cases:

```{code-cell} ipython3
def f(row):
    x = row['a']
    if x > 3:
            return x
    else:
            return 1.5

df = cudf.DataFrame({
    'a': [1, 3, 5]
})
df
```

```{code-cell} ipython3
df.apply(f, axis=1)
```

Any number of columns and many arithmetic operators are supported, allowing for complex UDFs:

```{code-cell} ipython3
def f(row):
    return row['a'] + (row['b'] - (row['c'] / row['d'])) % row['e']

df = cudf.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [cudf.NA, 4, 4],
    'd': [8, 7, 8],
    'e': [7, 1, 6]
})
df
```

```{code-cell} ipython3
df.apply(f, axis=1)
```

# Numba kernels for DataFrames

+++


We could apply a UDF on a DataFrame like we did above with `forall`. We'd need to write a kernel that expects multiple inputs, and pass multiple Series as arguments when we execute our kernel. Because this is fairly common and can be difficult to manage, cuDF provides two APIs to streamline this: `apply_rows` and `apply_chunks`. Below, we walk through an example of using `apply_rows`. `apply_chunks` works in a similar way, but also offers more control over low-level kernel behavior.

Now that we have two numeric columns in our DataFrame, let's write a kernel that uses both of them.

```{code-cell} ipython3
def conditional_add(x, y, out):
    for i, (a, e) in enumerate(zip(x, y)):
        if a > 0:
            out[i] = a + e
        else:
            out[i] = a
```

Notice that we need to `enumerate` through our `zipped` function arguments (which either match or are mapped to our input column names). We can pass this kernel to `apply_rows`. We'll need to specify a few arguments:
- incols
    - A list of names of input columns that match the function arguments. Or, a dictionary mapping input column names to their corresponding function arguments such as `{'col1': 'arg1'}`.
- outcols
    - A dictionary defining our output column names and their data types. These names must match our function arguments.
- kwargs (optional)
    - We can optionally pass keyword arguments as a dictionary. Since we don't need any, we pass an empty one.
    
While it looks like our function is looping sequentially through our columns, it actually executes in parallel in multiple threads on the GPU. This parallelism is the heart of GPU-accelerated computing. With that background, we're ready to use our UDF.

```{code-cell} ipython3
df = df.apply_rows(conditional_add, 
                   incols={'a':'x', 'e':'y'},
                   outcols={'out': np.float64},
                   kwargs={}
                  )
df.head()
```

As expected, we see our conditional addition worked. At this point, we've successfully executed UDFs on the core data structures of cuDF.

+++

## Null Handling in `apply_rows` and `apply_chunks`

By default, DataFrame methods for applying UDFs like `apply_rows` will handle nulls pessimistically (all rows with a null value will be removed from the output if they are used in the kernel). Exploring how not handling not pessimistically can lead to undefined behavior is outside the scope of this guide. Suffice it to say, pessimistic null handling is the safe and consistent approach. You can see an example below.

```{code-cell} ipython3
def gpu_add(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = x + y

df = randomdata(nrows=5, dtypes={'a':int, 'b':int, 'c':int}, seed=12)
df.loc[2, 'a'] = None
df.loc[3, 'b'] = None
df.loc[1, 'c'] = None
df.head()
```

In the dataframe above, there are three null values. Each column has a null in a different row. When we use our UDF with `apply_rows`, our output should have two nulls due to pessimistic null handling (because we're not using column `c`, the null value there does not matter to us).

```{code-cell} ipython3
df = df.apply_rows(gpu_add, 
              incols=['a', 'b'],
              outcols={'out':np.float64},
              kwargs={})
df.head()
```

As expected, we end up with two nulls in our output. The null values from the columns we used propogated to our output, but the null from the column we ignored did not.

+++

## Rolling Window UDFs

For time-series data, we may need to operate on a small \"window\" of our column at a time, processing each portion independently. We could slide (\"roll\") this window over the entire column to answer questions like \"What is the 3-day moving average of a stock price over the past year?"

We can apply more complex functions to rolling windows to `rolling` Series and DataFrames using `apply`. This example is adapted from cuDF's [API documentation](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.rolling.html). First, we'll create an example Series and then create a `rolling` object from the Series.

```{code-cell} ipython3
ser = cudf.Series([16, 25, 36, 49, 64, 81], dtype='float64')
ser
```

```{code-cell} ipython3
rolling = ser.rolling(window=3, min_periods=3, center=False)
rolling
```

Next, we'll define a function to use on our rolling windows. We created this one to highlight how you can include things like loops, mathematical functions, and conditionals. Rolling window UDFs do not yet support null values.

```{code-cell} ipython3
import math

def example_func(window):
    b = 0
    for a in window:
        b = max(b, math.sqrt(a))
    if b == 8:
        return 100    
    return b
```

We can execute the function by passing it to `apply`. With `window=3`, `min_periods=3`, and `center=False`, our first two values are `null`.

```{code-cell} ipython3
rolling.apply(example_func)
```

We can apply this function to every column in a DataFrame, too.

```{code-cell} ipython3
df2 = cudf.DataFrame()
df2['a'] = np.arange(55, 65, dtype='float64')
df2['b'] = np.arange(55, 65, dtype='float64')
df2.head()
```

```{code-cell} ipython3
rolling = df2.rolling(window=3, min_periods=3, center=False)
rolling.apply(example_func)
```

## GroupBy DataFrame UDFs

We can also apply UDFs to grouped DataFrames using `apply_grouped`. This example is also drawn and adapted from the RAPIDS [API documentation]().

First, we'll group our DataFrame based on column `b`, which is either True or False.

```{code-cell} ipython3
df = randomdata(nrows=10, dtypes={'a':float, 'b':bool, 'c':str, 'e': float}, seed=12)
df.head()
```

```{code-cell} ipython3
grouped = df.groupby(['b'])
```

Next we'll define a function to apply to each group independently. In this case, we'll take the rolling average of column `e`, and call that new column `rolling_avg_e`.

```{code-cell} ipython3
def rolling_avg(e, rolling_avg_e):
    win_size = 3
    for i in range(cuda.threadIdx.x, len(e), cuda.blockDim.x):
        if i < win_size - 1:
            # If there is not enough data to fill the window,
            # take the average to be NaN
            rolling_avg_e[i] = np.nan
        else:
            total = 0
            for j in range(i - win_size + 1, i + 1):
                total += e[j]
            rolling_avg_e[i] = total / win_size
```

We can execute this with a very similar API to `apply_rows`. This time, though, it's going to execute independently for each group.

```{code-cell} ipython3
results = grouped.apply_grouped(rolling_avg,
                               incols=['e'],
                               outcols=dict(rolling_avg_e=np.float64))
results
```

Notice how, with a window size of three in the kernel, the first two values in each group for our output column are null.

+++

## Numba Kernels on CuPy Arrays

We can also execute Numba kernels on CuPy NDArrays, again thanks to the `__cuda_array_interface__`. We can even run the same UDF on the Series and the CuPy array. First, we define a Series and then create a CuPy array from that Series.

```{code-cell} ipython3
import cupy as cp

s = cudf.Series([1.0, 2, 3, 4, 10])
arr = cp.asarray(s)
arr
```

Next, we define a UDF and execute it on our Series. We need to allocate a Series of the same size for our output, which we'll call `out`.

```{code-cell} ipython3
@cuda.jit
def multiply_by_5(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] * 5
        
out = cudf.Series(cp.zeros(len(s), dtype='int32'))
multiply_by_5.forall(s.shape[0])(s, out)
out
```

Finally, we execute the same function on our array. We allocate an empty array `out` to store our results.

```{code-cell} ipython3
out = cp.empty_like(arr)
multiply_by_5.forall(arr.size)(arr, out)
out
```

## Caveats

+++

- Only numeric nondecimal scalar types are currently supported as of yet, but strings and structured types are in planning. Attempting to use this API with those types will throw a `TypeError`.
- We do not yet fully support all arithmetic operators. Certain ops like bitwise operations are not currently implemented, but planned in future releases. If an operator is needed, a github issue should be raised so that it can be properly prioritized and implemented.

+++

## Summary

This guide has covered a lot of content. At this point, you should hopefully feel comfortable writing UDFs (with or without null values) that operate on

- Series
- DataFrame
- Rolling Windows
- GroupBy DataFrames
- CuPy NDArrays
- Numba DeviceNDArrays
- Generalized NA UDFs


For more information please see the [cuDF](https://docs.rapids.ai/api/cudf/nightly/), [Numba.cuda](https://numba.pydata.org/numba-doc/dev/cuda/index.html), and [CuPy](https://docs-cupy.chainer.org/en/stable/) documentation.
