# Copyright (c) 2024, NVIDIA CORPORATION.
import urllib

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_url_encode():
    data = ["/home/nfs", None]
    arr = pa.array(data)
    result = plc.strings.convert.convert_urls.url_encode(
        plc.interop.from_arrow(arr)
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
        plc.interop.from_arrow(arr)
    )
    expected = pa.array(
        [
            urllib.parse.unquote(url) if isinstance(url, str) else url
            for url in data
        ]
    )
    assert_column_eq(result, expected)
