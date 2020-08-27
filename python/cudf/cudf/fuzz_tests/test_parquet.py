import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("data", [[1], [2]])
@pytest.mark.parametrize("data2", [[1], [2]])
def test_are_equal_custom(data, data2):
    assert data == data2


@pytest.mark.parametrize("seed", [0, 10, 100])
@pytest.mark.parametrize("size", [10, 100, 0, 100000])
@pytest.mark.parametrize("int_null_frequency", [0.1, 0, 1.0, 0.7])
@pytest.mark.parametrize("str_null_frequency", [0.1, 0, 1.0, 0.7])
def test_write_to_parquet(
    tmpdir, seed, size, int_null_frequency, str_null_frequency
):
    df = cudf.DataFrame(
        {
            "int_col": cudf.datasets.get_rand_int(
                rows=size, seed=seed, null_frequency=int_null_frequency
            ),
            "str_col": cudf.datasets.get_rand_str(
                rows=size, seed=seed, null_frequency=str_null_frequency
            ),
        }
    )
    fname = tmpdir.join("test_pq_test.parquet")
    df.to_parquet(fname)

    pdf = pd.read_parquet(fname)
    assert_eq(df, pdf)
