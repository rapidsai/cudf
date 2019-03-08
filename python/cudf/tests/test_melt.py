import pytest
import numpy as np
import pandas as pd
from cudf import melt as cudf_melt
from cudf.dataframe import DataFrame


@pytest.mark.parametrize('num_id_vars', [0, 1, 2, 10])
@pytest.mark.parametrize('num_value_vars', [0, 1, 2, 10])
@pytest.mark.parametrize('num_rows', [1, 2, 1000])
@pytest.mark.parametrize(
    'dtype',
    ['int8', 'int16', 'int32', 'int64', 'float32', 'float64', 'datetime64[ms]']
)
@pytest.mark.parametrize('nulls', ['none', 'some', 'all'])
def test_melt(nulls, num_id_vars, num_value_vars, num_rows, dtype):
    if dtype not in ['float32', 'float64'] and nulls in ['some', 'all']:
        pytest.skip(msg='nulls not supported in dtype: ' + dtype)

    pdf = pd.DataFrame()
    id_vars = []
    for i in range(num_id_vars):
        colname = 'id' + str(i)
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == 'some':
            idx = np.random.choice(num_rows,
                                   size=int(num_rows/2),
                                   replace=False)
            data[idx] = np.nan
        elif nulls == 'all':
            data[:] = np.nan
        pdf[colname] = data
        id_vars.append(colname)

    value_vars = []
    for i in range(num_value_vars):
        colname = 'val' + str(i)
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == 'some':
            idx = np.random.choice(num_rows,
                                   size=int(num_rows/2),
                                   replace=False)
            data[idx] = np.nan
        elif nulls == 'all':
            data[:] = np.nan
        pdf[colname] = data
        value_vars.append(colname)

    gdf = DataFrame.from_pandas(pdf)

    got = cudf_melt(frame=gdf, id_vars=id_vars, value_vars=value_vars)

    expect = pd.melt(frame=pdf, id_vars=id_vars, value_vars=value_vars)
    # pandas' melt makes the 'variable' column of 'object' type (string)
    # cuDF's melt makes it Categorical because it doesn't support strings
    expect['variable'] = expect['variable'].astype('category')

    pd.testing.assert_frame_equal(
        expect,
        got.to_pandas()
    )
