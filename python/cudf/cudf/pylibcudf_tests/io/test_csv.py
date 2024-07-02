# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pytest
from utils import COMPRESSION_TYPE_TO_PANDAS, assert_table_and_meta_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType


# TODO: de-dupe with make_json_source
def make_csv_source(path_or_buf, pa_table, **kwargs):
    """
    Uses pandas to write a pyarrow Table to a JSON file.
    The caller is responsible for making sure that no arguments
    unsupported by pandas are passed in.
    """
    df = pa_table.to_pandas()
    mode = "w"
    if "compression" in kwargs:
        kwargs["compression"] = COMPRESSION_TYPE_TO_PANDAS[
            kwargs["compression"]
        ]
        if kwargs["compression"] is not None:
            mode = "wb"
    df.to_csv(path_or_buf, index=False, mode=mode, **kwargs)
    # print(df.to_csv(index=False, mode=mode, **kwargs))
    if isinstance(path_or_buf, io.IOBase):
        path_or_buf.seek(0)
    return path_or_buf


def test_read_csv_basic(table_data, source_or_sink, compression_type):
    if compression_type in {
        # Not supported by libcudf
        CompressionType.SNAPPY,
        CompressionType.XZ,
        CompressionType.ZSTD,
        # Not supported by pandas
        # TODO: find a way to test these
        CompressionType.BROTLI,
        CompressionType.LZ4,
        CompressionType.LZO,
        CompressionType.ZLIB,
    }:
        pytest.skip("unsupported compression type by pandas/libcudf")

    _, pa_table = table_data

    # can't compress non-binary data with pandas
    if isinstance(source_or_sink, io.StringIO):
        compression_type = CompressionType.NONE

    if len(pa_table) > 0:
        # Drop the string column for now, since it contains ints
        # (which won't roundtrip since they are not quoted by python csv writer)
        # also uint64 will get confused for int64
        pa_table = pa_table.drop_columns(
            [
                "col_string",
                "col_uint64",
                # Nested types don't work by default
                "col_list<item: int64>",
                "col_list<item: list<item: int64>>",
                "col_struct<v: int64 not null>",
                "col_struct<a: int64 not null, b_struct: struct<b: double not null> not null>",
            ]
        )

    source = make_csv_source(
        source_or_sink, pa_table, compression=compression_type
    )

    if isinstance(source, io.StringIO):
        pytest.skip("todo: something going wrong investigate!")
    res = plc.io.csv.read_csv(
        plc.io.SourceInfo([source]),
        compression=compression_type,
    )

    assert_table_and_meta_eq(
        pa_table, res, check_types=False if len(pa_table) == 0 else True
    )
