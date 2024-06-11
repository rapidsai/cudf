# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pytest
from utils import COMPRESSION_TYPE_TO_PANDAS, assert_table_and_metas_eq

import cudf._lib.pylibcudf as plc


def make_json_source(path_or_buf, pa_table, **kwargs):
    """
    Uses pandas to write a pyarrow Table to a JSON file.

    The caller is responsible for making sure that no arguments
    unsupported by pandas are passed in.
    """
    df = pa_table.to_pandas()
    if "compression" in kwargs:
        kwargs["compression"] = COMPRESSION_TYPE_TO_PANDAS[
            kwargs["compression"]
        ]
    df.to_json(path_or_buf, orient="records", **kwargs)
    if isinstance(path_or_buf, io.IOBase):
        path_or_buf.seek(0)
    return path_or_buf


@pytest.mark.parametrize("lines", [True, False])
def test_read_json_basic(table_data, source_or_sink, lines, compression_type):
    exp, pa_table = table_data
    source = make_json_source(
        source_or_sink,
        pa_table,
        lines=lines,
    )
    if isinstance(source, io.IOBase):
        source.seek(0)

    res = plc.io.json.read_json(
        plc.io.SourceInfo([source]),
        compression=compression_type,
        lines=lines,
    )
    print(exp.tbl)
    print(res.tbl)
    assert_table_and_metas_eq(exp, res)
    # processed_dtypes,
    # byte_range_offset = byte_range[0] if byte_range is not None else 0,
    # byte_range_size = byte_range[1] if byte_range is not None else 0,
    # keep_quotes = keep_quotes,
    # mixed_types_as_string = mixed_types_as_string,
    # prune_columns = prune_columns,
    # recovery_mode = _get_json_recovery_mode(on_bad_lines)
