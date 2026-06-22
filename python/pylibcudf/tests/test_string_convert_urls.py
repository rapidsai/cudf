# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import urllib

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_url_encode():
    data = ["/home/nfs", None]
    arr = pa.array(data)
    result = plc.strings.convert.convert_urls.url_encode(
        plc.Column.from_arrow(arr)
    )
    expected = pa.array(
        [
            urllib.parse.quote(url, safe="") if isinstance(url, str) else url
            for url in data
        ]
    )
    assert_column_eq(result, expected)


def test_url_decode():
    data = ["%2Fhome%2fnfs", None]
    arr = pa.array(data)
    result = plc.strings.convert.convert_urls.url_decode(
        plc.Column.from_arrow(arr)
    )
    expected = pa.array(
        [
            urllib.parse.unquote(url) if isinstance(url, str) else url
            for url in data
        ]
    )
    assert_column_eq(result, expected)
