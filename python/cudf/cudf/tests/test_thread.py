# Copyright (c) 2025, NVIDIA CORPORATION.

import cudf


def test_thread():
    expected_settings = [1, 4, 16, 4, 1]

    for expected_setting in expected_settings:
        cudf.io.set_num_io_threads(expected_setting)
        assert expected_setting == cudf.io.num_io_threads()
