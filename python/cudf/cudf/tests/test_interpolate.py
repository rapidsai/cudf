import pandas as pd
import cudf
import pytest
from cudf.testing._utils import assert_eq

@pytest.mark.parametrize("data", [
    {
        'A': [1, None, 3]
    },
    {
        'A': [None, 2, 3, None, 5]
    },
    {
        'A': [1, None, 3],
        'B': [None, 2, 3]
    }
])
@pytest.mark.parametrize("method", ['linear'])
@pytest.mark.parametrize("axis", [0])
def test_interpolate_nans(data, method,axis):  
    # doesn't seem to work with NAs just yet
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    
    expect = pdf.interpolate(method=method, axis=axis)
    got = gdf.interpolate(method=method, axis=axis)
    breakpoint()
    assert_eq(expect, got)
