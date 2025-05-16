# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    return pa.array(
        [
            "e n",
            "i t",
            "i s",
            "e s",
            "en t",
            "c e",
            "es t",
            "en ce",
            "t est",
            "s ent",
        ]
    )


@pytest.mark.parametrize(
    "separator", [None, plc.interop.from_arrow(pa.scalar("e"))]
)
def test_byte_pair_encoding(input_col, separator):
    plc_col = plc.interop.from_arrow(
        pa.array(["test sentence", "thisis test"])
    )
    result = plc.nvtext.byte_pair_encode.byte_pair_encoding(
        plc_col,
        plc.nvtext.byte_pair_encode.BPEMergePairs(
            plc.interop.from_arrow(input_col)
        ),
        separator,
    )
    if separator is None:
        expected = pa.array(["test   sent ence", "t h is is   test"])
    else:
        expected = pa.array(["teste esenteence", "teheiseise etest"])
    assert_column_eq(result, expected)
