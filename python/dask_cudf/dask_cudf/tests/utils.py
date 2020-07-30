import numpy as np
import pandas as pd
import pandas.util.testing as tm

from dask.dataframe.utils import (
    _check_dask,
    _maybe_sort,
    assert_divisions,
    assert_sane_keynames,
)


def assert_dd_eq(
    a,
    b,
    check_names=True,
    check_dtypes=True,
    check_divisions=True,
    check_index=True,
    **kwargs,
):
    if check_divisions:
        assert_divisions(a)
        assert_divisions(b)
        if hasattr(a, "divisions") and hasattr(b, "divisions"):
            at = type(np.asarray(a.divisions).tolist()[0])  # numpy to python
            bt = type(np.asarray(b.divisions).tolist()[0])  # scalar conversion
            assert at == bt, (at, bt)
    assert_sane_keynames(a)
    assert_sane_keynames(b)
    a = _check_dask(a, check_names=check_names, check_dtypes=check_dtypes)
    b = _check_dask(b, check_names=check_names, check_dtypes=check_dtypes)
    if not check_index:
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
    if hasattr(a, "to_pandas"):
        try:
            a = a.to_pandas(nullable_pd_dtype=False)
        except TypeError:
            a = a.to_pandas()
    if hasattr(b, "to_pandas"):
        try:
            b = b.to_pandas(nullable_pd_dtype=False)
        except TypeError:
            b = b.to_pandas()
    if isinstance(a, pd.DataFrame):
        a = _maybe_sort(a)
        b = _maybe_sort(b)
        tm.assert_frame_equal(a, b, **kwargs)
    elif isinstance(a, pd.Series):
        a = _maybe_sort(a)
        b = _maybe_sort(b)
        tm.assert_series_equal(a, b, check_names=check_names, **kwargs)
    elif isinstance(a, pd.Index):
        tm.assert_index_equal(a, b, **kwargs)
    else:
        if a == b:
            return True
        else:
            if np.isnan(a):
                assert np.isnan(b)
            else:
                assert np.allclose(a, b)
    return True
