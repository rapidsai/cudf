# Frequently Asked Questions


## Third Party Integrations

| Library          | Status |
|------------------|--------|
| cuGraph          | ✅      |
| cuML             | ✅      |
| Dask             | ✅      |
| Matplotlib       | ✅      |
| NumPy            | ✅      |
| Pandas Profiling | ❌      |
| PyCaret          | ❌      |
| PyTorch          | ✅      |
| Scikit-Learn     | ✅      |
| SciPy            | ✅      |
| Seaborn          | ✅      |
| Stumpy           | ✅      |
| Tensorflow       | ✅      |
| XGBoost          | ✅      |



## Known limitations
- Currently, cudf.pandas is in open beta and while our goal is 100% passing coverage for the pandas test suit, we are currently passing 90.1% of the pandas test suite.  However, we are contintinually improving moving towards 100% passing coverage.  Our coverage falls short around NaN vs Null Behavior, Datetime handling, and Nullable/Arrow/Extension types
- Using cudf.pandas with ``torch.from_numpy(df.values)`` -- though ``torch.tesnor(df.values``) works. xref: https://github.com/rapidsai/xdf/issues/210
- Using the Profiler (link to proflier section)  may not work while using third-party
  libraries. xref: https://github.com/rapidsai/xdf/issues/331


## Profiler Details
