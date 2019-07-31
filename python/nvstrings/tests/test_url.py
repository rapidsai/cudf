# Copyright (c) 2019, NVIDIA CORPORATION.

import urllib.parse

import nvstrings
from utils import assert_eq

urls1 = ["http://www.hellow.com", "/home/nvidia/nfs", "123.45 ~ABCDEF"]
urls2 = [
    "http://www.hellow.com?k1=acc%C3%A9nted&k2=a%2F/b.c",
    "%2Fhome%2fnfs",
    "987%20ZYX",
]


def test_encode_url():
    s = nvstrings.to_device(urls1)
    got = s.url_encode()
    expected = []
    for url in urls1:
        expected.append(urllib.parse.quote(url, safe="~"))
    assert_eq(got, expected)


def test_decode_url():
    s = nvstrings.to_device(urls2)
    got = s.url_decode()
    expected = []
    for url in urls2:
        expected.append(urllib.parse.unquote(url))
    assert_eq(got, expected)
