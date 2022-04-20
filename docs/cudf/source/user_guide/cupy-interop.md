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

# Interoperability of cuDF with CuPy

This notebook provides introductory examples of how you can use cuDF and CuPy together to take advantage of CuPy array functionality (such as advanced linear algebra operations).

```{code-cell} ipython3
import timeit
from packaging import version

import cupy as cp
import cudf

if version.parse(cp.__version__) >= version.parse("10.0.0"):
    cupy_from_dlpack = cp.from_dlpack
else:
    cupy_from_dlpack = cp.fromDlpack
```

### Converting a cuDF DataFrame to a CuPy Array

If we want to convert a cuDF DataFrame to a CuPy ndarray, There are multiple ways to do it:

1. We can use the [dlpack](https://github.com/dmlc/dlpack) interface.

2. We can also use `DataFrame.values`.

3. We can also convert via the [CUDA array interface](https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html) by using cuDF's `as_gpu_matrix` and CuPy's `asarray` functionality.

```{code-cell} ipython3
nelem = 10000
df = cudf.DataFrame({'a':range(nelem),
                     'b':range(500, nelem + 500),
                     'c':range(1000, nelem + 1000)}
                   )

%timeit arr_cupy = cupy_from_dlpack(df.to_dlpack())
%timeit arr_cupy = df.values
%timeit arr_cupy = df.to_cupy()
```

```{code-cell} ipython3
arr_cupy = cupy_from_dlpack(df.to_dlpack())
arr_cupy
```

### Converting a cuDF Series to a CuPy Array

+++

There are also multiple ways to convert a cuDF Series to a CuPy array:

1. We can pass the Series to `cupy.asarray` as cuDF Series exposes [`__cuda_array_interface__`](https://docs-cupy.chainer.org/en/stable/reference/interoperability.html).
2. We can leverage the dlpack interface `to_dlpack()`. 
3. We can also use `Series.values` 

```{code-cell} ipython3
col = 'a'

%timeit cola_cupy = cp.asarray(df[col])
%timeit cola_cupy = cupy_from_dlpack(df[col].to_dlpack())
%timeit cola_cupy = df[col].values
```

```{code-cell} ipython3
cola_cupy = cp.asarray(df[col])
cola_cupy
```

From here, we can proceed with normal CuPy workflows, such as reshaping the array, getting the diagonal, or calculating the norm.

```{code-cell} ipython3
reshaped_arr = cola_cupy.reshape(50, 200)
reshaped_arr
```

```{code-cell} ipython3
reshaped_arr.diagonal()
```

```{code-cell} ipython3
cp.linalg.norm(reshaped_arr)
```

### Converting a CuPy Array to a cuDF DataFrame

We can also convert a CuPy ndarray to a cuDF DataFrame. Like before, there are multiple ways to do it:

1. **Easiest;** We can directly use the `DataFrame` constructor.

2. We can use CUDA array interface with the `DataFrame` constructor.

3. We can also use the [dlpack](https://github.com/dmlc/dlpack) interface.

For the latter two cases, we'll need to make sure that our CuPy array is Fortran contiguous in memory (if it's not already). We can either transpose the array or simply coerce it to be Fortran contiguous beforehand.

```{code-cell} ipython3
%timeit reshaped_df = cudf.DataFrame(reshaped_arr)
```

```{code-cell} ipython3
reshaped_df = cudf.DataFrame(reshaped_arr)
reshaped_df.head()
```

We can check whether our array is Fortran contiguous by using cupy.isfortran or looking at the [flags](https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.flags) of the array.

```{code-cell} ipython3
cp.isfortran(reshaped_arr)
```

In this case, we'll need to convert it before going to a cuDF DataFrame. In the next two cells, we create the DataFrame by leveraging dlpack and the CUDA array interface, respectively.

```{code-cell} ipython3
%%timeit

fortran_arr = cp.asfortranarray(reshaped_arr)
reshaped_df = cudf.DataFrame(fortran_arr)
```

```{code-cell} ipython3
%%timeit

fortran_arr = cp.asfortranarray(reshaped_arr)
reshaped_df = cudf.from_dlpack(fortran_arr.toDlpack())
```

```{code-cell} ipython3
fortran_arr = cp.asfortranarray(reshaped_arr)
reshaped_df = cudf.DataFrame(fortran_arr)
reshaped_df.head()
```

### Converting a CuPy Array to a cuDF Series

To convert an array to a Series, we can directly pass the array to the `Series` constructor.

```{code-cell} ipython3
cudf.Series(reshaped_arr.diagonal()).head()
```

### Interweaving CuDF and CuPy for Smooth PyData Workflows

RAPIDS libraries and the entire GPU PyData ecosystem are developing quickly, but sometimes a one library may not have the functionality you need. One example of this might be taking the row-wise sum (or mean) of a Pandas DataFrame. cuDF's support for row-wise operations isn't mature, so you'd need to either transpose the DataFrame or write a UDF and explicitly calculate the sum across each row. Transposing could lead to hundreds of thousands of columns (which cuDF wouldn't perform well with) depending on your data's shape, and writing a UDF can be time intensive.

By leveraging the interoperability of the GPU PyData ecosystem, this operation becomes very easy. Let's take the row-wise sum of our previously reshaped cuDF DataFrame.

```{code-cell} ipython3
reshaped_df.head()
```

We can just transform it into a CuPy array and use the `axis` argument of `sum`.

```{code-cell} ipython3
new_arr = cupy_from_dlpack(reshaped_df.to_dlpack())
new_arr.sum(axis=1)
```

With just that single line, we're able to seamlessly move between data structures in this ecosystem, giving us enormous flexibility without sacrificing speed.

+++

### Converting a cuDF DataFrame to a CuPy Sparse Matrix

We can also convert a DataFrame or Series to a CuPy sparse matrix. We might want to do this if downstream processes expect CuPy sparse matrices as an input.

The sparse matrix data structure is defined by three dense arrays. We'll define a small helper function for cleanliness.

```{code-cell} ipython3
def cudf_to_cupy_sparse_matrix(data, sparseformat='column'):
    """Converts a cuDF object to a CuPy Sparse Column matrix.
    """
    if sparseformat not in ('row', 'column',):
        raise ValueError("Let's focus on column and row formats for now.")
    
    _sparse_constructor = cp.sparse.csc_matrix
    if sparseformat == 'row':
        _sparse_constructor = cp.sparse.csr_matrix

    return _sparse_constructor(cp.from_dlpack(data.to_dlpack()))
```

We can define a sparsely populated DataFrame to illustrate this conversion to either sparse matrix format.

```{code-cell} ipython3
df = cudf.DataFrame()
nelem = 10000
nonzero = 1000
for i in range(20):
    arr = cp.random.normal(5, 5, nelem)
    arr[cp.random.choice(arr.shape[0], nelem-nonzero, replace=False)] = 0
    df['a' + str(i)] = arr
```

```{code-cell} ipython3
df.head()
```

```{code-cell} ipython3
sparse_data = cudf_to_cupy_sparse_matrix(df)
print(sparse_data)
```

From here, we could continue our workflow with a CuPy sparse matrix.

For a full list of the functionality built into these libraries, we encourage you to check out the API docs for [cuDF](https://docs.rapids.ai/api/cudf/nightly/) and [CuPy](https://docs-cupy.chainer.org/en/stable/index.html).
