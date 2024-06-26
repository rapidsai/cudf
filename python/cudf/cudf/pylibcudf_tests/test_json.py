# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pandas as pd
import pyarrow as pa
import pytest
from utils import COMPRESSION_TYPE_TO_PANDAS, assert_table_and_meta_eq

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


def write_json_bytes(source, json_bytes):
    if not isinstance(source, io.IOBase):
        with open(source, "wb") as source_f:
            source_f.write(json_bytes)
    else:
        source.write(json_bytes)
        source.seek(0)


@pytest.mark.parametrize("lines", [True, False])
def test_read_json_basic(table_data, source_or_sink, lines, compression_type):
    # TODO: separate sources and sinks fixtures?
    # alternately support StringIO as a source in SourceInfo
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("skip StringIO since it is only a valid sink")
    # if compression_type in {plc.io.types.CompressionType.SNAPPY,
    #                         #plc.io.types.CompressionType.GZIP,
    #                         #plc.io.types.CompressionType.BZIP2,
    #                         #plc.io.types.CompressionType.BROTLI
    #                         }:
    #     pytest.skip("unsupported libcudf compression type")
    _, pa_table = table_data
    source = make_json_source(
        source_or_sink,
        pa_table,
        lines=lines,
    )
    if isinstance(source, io.IOBase):
        source.seek(0)

    res = plc.io.json.read_json(
        plc.io.SourceInfo([source]),
        # TODO: re-enable me!
        # compression=compression_type,
        lines=lines,
    )

    # orient=records is lossy
    # and doesn't preserve column names when there's zero rows in the table
    if len(pa_table) == 0:
        pa_table = pa.table([])

    # Convert non-nullable struct fields to nullable fields
    # since nullable=False cannot roundtrip through orient='records'
    # JSON format
    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


@pytest.mark.parametrize("chunk_size", [10, 15, 20])
def test_read_json_lines_byte_range(source_or_sink, chunk_size):
    source = source_or_sink
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("skip StringIO since it is only a valid sink")

    json_bytes = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n".encode("utf-8")
    write_json_bytes(source, json_bytes)

    tbls_w_meta = []
    for chunk_start in range(0, len(json_bytes), chunk_size):
        tbls_w_meta.append(
            plc.io.json.read_json(
                plc.io.SourceInfo([source]),
                lines=True,
                byte_range_offset=chunk_start,
                byte_range_size=chunk_start + chunk_size,
            )
        )

    if isinstance(source, io.IOBase):
        source.seek(0)
    exp = pd.read_json(source, orient="records", lines=True)

    # TODO: can do this operation using pylibcudf
    tbls = []
    for tbl_w_meta in tbls_w_meta:
        if tbl_w_meta.tbl.num_rows() > 0:
            tbls.append(plc.interop.to_arrow(tbl_w_meta.tbl))
    full_tbl = pa.concat_tables(tbls)

    # Clobber the first table, since we don't have a way to transfer metadata
    # between tables
    # TODO: this can be better! maybe TableWithMetadata.metadata_like?
    first_tbl = tbls_w_meta[0]
    first_tbl.tbl = plc.interop.from_arrow(full_tbl)
    assert_table_and_meta_eq(pa.Table.from_pandas(exp), first_tbl)


@pytest.mark.parametrize("keep_quotes", [True, False])
def test_read_json_lines_keep_quotes(keep_quotes, source_or_sink):
    source = source_or_sink
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("skip StringIO since it is only a valid sink")

    json_bytes = '["a", "b", "c"]\n'.encode("utf-8")
    write_json_bytes(source, json_bytes)

    tbl_w_meta = plc.io.json.read_json(
        plc.io.SourceInfo([source]), lines=True, keep_quotes=keep_quotes
    )

    template = "{0}"
    if keep_quotes:
        template = '"{0}"'

    exp = pa.Table.from_arrays(
        [
            [template.format("a")],
            [template.format("b")],
            [template.format("c")],
        ],
        names=["0", "1", "2"],
    )

    assert_table_and_meta_eq(exp, tbl_w_meta)


@pytest.mark.parametrize(
    "recovery_mode", [opt for opt in plc.io.types.JSONRecoveryMode]
)
def test_read_json_lines_recovery_mode(recovery_mode, source_or_sink):
    source = source_or_sink
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("skip StringIO since it is only a valid sink")

    json_bytes = (
        '{"a":1,"b":10}\n{"a":2,"b":11}\nabc\n{"a":3,"b":12}\n'.encode("utf-8")
    )
    write_json_bytes(source, json_bytes)

    if recovery_mode == plc.io.types.JSONRecoveryMode.FAIL:
        with pytest.raises(RuntimeError):
            plc.io.json.read_json(
                plc.io.SourceInfo([source]),
                lines=True,
                recovery_mode=recovery_mode,
            )
    else:
        # Recover case (bad values replaced with nulls)
        tbl_w_meta = plc.io.json.read_json(
            plc.io.SourceInfo([source]),
            lines=True,
            recovery_mode=recovery_mode,
        )
        exp = pa.Table.from_arrays(
            [[1, 2, None, 3], [10, 11, None, 12]], names=["a", "b"]
        )
        assert_table_and_meta_eq(exp, tbl_w_meta)


# TODO: test for dtypes!


# TODO: Add tests for these!
# Tests were not added in the initial PR porting the JSON reader to pylibcudf
# to save time (and since there are no existing tests for these in Python cuDF)
# mixed_types_as_string = mixed_types_as_string,
# prune_columns = prune_columns,
