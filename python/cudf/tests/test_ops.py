
import pytest

import cudf
from cudf.tests.utils import assert_eq


def test_sqrt_float():
    assert cudf.sqrt(16.0) == 4.0
    assert_eq(cudf.sqrt(cudf.Series([4.0, 9, 16])), cudf.Series([2.0, 3, 4]))
    assert_eq(cudf.sqrt(cudf.DataFrame({'x': [4.0, 9, 16]})),
              cudf.DataFrame({'x': [2.0, 3, 4]}))


@pytest.mark.xfail(reason="integer sqrt results in GDF_UNSUPPORTED_DTYPE")
def test_sqrt_integer():
    assert cudf.sqrt(16) == 4
    assert_eq(cudf.sqrt(cudf.Series([4, 9, 16])), cudf.Series([2, 3, 4]))
    assert_eq(cudf.sqrt(cudf.DataFrame({'x': [4, 9, 16]})),
              cudf.DataFrame({'x': [2, 3, 4]}))
