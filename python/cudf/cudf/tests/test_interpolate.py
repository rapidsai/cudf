import pandas as pd
import cudf
import pytest
from cudf.testing._utils import assert_eq, assert_exceptions_equal

@pytest.mark.parametrize("data", [
    # basics
    {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    },
    {
        'A': [1, None, 3],
        'B': [4, None, 6]
    },
    {
        'A': [None, 2, 3],
        'B': [4, 5, None]
    }
])
@pytest.mark.parametrize("method", ['linear'])
@pytest.mark.parametrize("axis", [0])
def test_interpolate_dataframe(data, method, axis):  
    # doesn't seem to work with NAs just yet
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    
    expect = pdf.interpolate(method=method, axis=axis)
    got = gdf.interpolate(method=method, axis=axis)
    assert_eq(expect, got)

@pytest.mark.parametrize("data", [
    [1,2,3],
    [1, None, 3],
    [None, 2, None, 4],
    [1, None, 3, None],
    [0.1, 0.2, 0.3]
])
@pytest.mark.parametrize("method", ['linear'])
@pytest.mark.parametrize("axis", [0])
def test_interpolate_series(data, method, axis):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method=method, axis=axis)
    got = gsr.interpolate(method=method, axis=axis)

    assert_eq(expect, got)

@pytest.mark.parametrize('data,kwargs', [
    (
        {
            'A': ['a','b','c'],
            'B': ['d','e','f']
        },
        {'axis': 0, 'method': 'linear'},
    )
])
def test_interpolate_dataframe_error_cases(data, kwargs):
    gsr = cudf.DataFrame(data)
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        lfunc = psr.interpolate,
        rfunc = gsr.interpolate,
        lfunc_args_and_kwargs = (
            [],
            kwargs
        ),
        rfunc_args_and_kwargs = (
            [],
            kwargs
        )
    )
