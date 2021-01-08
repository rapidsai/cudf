# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf import DateOffset
import pytest
@pytest.mark.parametrize('period', [1.5, 0.5, 'string', '1', '1.0'])
@pytest.mark.parametrize('freq', ['years', 'months'])
def test_construction_invalid(period, freq):
    kwargs = {freq: period}
    with pytest.raises(ValueError):
        DateOffset(**kwargs)
