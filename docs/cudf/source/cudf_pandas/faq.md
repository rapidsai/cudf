# Frequently Asked Questions

## How Closely Does This Match Pandas?

Currently, ``cudf.pandas`` is in open beta, and while our goal is 100% passing
coverage for the pandas test suite, we are currently passing 90.22% of the
pandas test suite.

We’re continually working to expand our passing coverage with a goal of 100%.
Our coverage falls short primarily around NaN vs Null behavior, Datetime
handling, and Nullable/Arrow/Extension types.



## Third Party Integration

Explain how thirdparty support works / isn’t guaranteed but probably works. We
explicitly test the following integrations and are always adding more!

Add call to action about telling us new libraries / if you’d like your library
tested.


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


##At-Scale Computing

This isn’t the tool for you!
CTA: Link out to Dask and Spark and describe value prop.


## Known limitations

- Using cudf.pandas with ``torch.from_numpy(df.values)`` -- though ``torch.tesnor(df.values``) works. xref: https://github.com/rapidsai/xdf/issues/210
- Using the Profiler (link to proflier section)  may not work while using third-party
  libraries. xref: https://github.com/rapidsai/xdf/issues/331

