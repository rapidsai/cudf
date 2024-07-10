# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pandas as pd
import pyarrow as pa
import pytest
from utils import (
    COMPRESSION_TYPE_TO_PANDAS,
    assert_table_and_meta_eq,
    sink_to_str,
)

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.io.types import CompressionType


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


def write_json_bytes(source, json_str):
    """
    Write a JSON string to the source
    """
    if not isinstance(source, io.IOBase):
        with open(source, "w") as source_f:
            source_f.write(json_str)
    else:
        if isinstance(source, io.BytesIO):
            json_str = json_str.encode("utf-8")
        source.write(json_str)
        source.seek(0)


@pytest.mark.parametrize("rows_per_chunk", [8, 100])
@pytest.mark.parametrize("lines", [True, False])
def test_write_json_basic(table_data, source_or_sink, lines, rows_per_chunk):
    plc_table_w_meta, pa_table = table_data
    sink = source_or_sink

    plc.io.json.write_json(
        plc.io.SinkInfo([sink]),
        plc_table_w_meta,
        lines=lines,
        rows_per_chunk=rows_per_chunk,
    )

    exp = pa_table.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)

    pd_result = exp.to_json(orient="records", lines=lines)

    assert str_result == pd_result


@pytest.mark.parametrize("include_nulls", [True, False])
@pytest.mark.parametrize("na_rep", ["null", "awef", ""])
def test_write_json_nulls(na_rep, include_nulls):
    names = ["a", "b"]
    pa_tbl = pa.Table.from_arrays(
        [pa.array([1.0, 2.0, None]), pa.array([True, None, False])],
        names=names,
    )
    plc_tbl = plc.interop.from_arrow(pa_tbl)
    plc_tbl_w_meta = plc.io.types.TableWithMetadata(
        plc_tbl, column_names=[(name, []) for name in names]
    )

    sink = io.StringIO()

    plc.io.json.write_json(
        plc.io.SinkInfo([sink]),
        plc_tbl_w_meta,
        na_rep=na_rep,
        include_nulls=include_nulls,
    )

    exp = pa_tbl.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)
    pd_result = exp.to_json(orient="records")

    if not include_nulls:
        # No equivalent in pandas, so we just
        # sanity check by making sure na_rep
        # doesn't appear in the output

        # don't quote null
        for name in names:
            assert f'{{"{name}":{na_rep}}}' not in str_result
        return

    # pandas doesn't suppport na_rep
    # let's just manually do str.replace
    pd_result = pd_result.replace("null", na_rep)

    assert str_result == pd_result


@pytest.mark.parametrize("true_value", ["True", "correct"])
@pytest.mark.parametrize("false_value", ["False", "wrong"])
def test_write_json_bool_opts(true_value, false_value):
    names = ["a"]
    pa_tbl = pa.Table.from_arrays([pa.array([True, None, False])], names=names)
    plc_tbl = plc.interop.from_arrow(pa_tbl)
    plc_tbl_w_meta = plc.io.types.TableWithMetadata(
        plc_tbl, column_names=[(name, []) for name in names]
    )

    sink = io.StringIO()

    plc.io.json.write_json(
        plc.io.SinkInfo([sink]),
        plc_tbl_w_meta,
        include_nulls=True,
        na_rep="null",
        true_value=true_value,
        false_value=false_value,
    )

    exp = pa_tbl.to_pandas()

    # Convert everything to string to make
    # comparisons easier
    str_result = sink_to_str(sink)
    pd_result = exp.to_json(orient="records")

    # pandas doesn't suppport na_rep
    # let's just manually do str.replace
    if true_value != "true":
        pd_result = pd_result.replace("true", true_value)
    if false_value != "false":
        pd_result = pd_result.replace("false", false_value)

    assert str_result == pd_result


@pytest.mark.parametrize("lines", [True, False])
def test_read_json_basic(
    table_data, source_or_sink, lines, compression_type, request
):
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

    # can't compress non-binary data with pandas
    if isinstance(source_or_sink, io.StringIO):
        compression_type = CompressionType.NONE

    _, pa_table = table_data

    source = make_json_source(
        source_or_sink, pa_table, lines=lines, compression=compression_type
    )

    request.applymarker(
        pytest.mark.xfail(
            condition=(
                len(pa_table) > 0
                and compression_type
                not in {CompressionType.NONE, CompressionType.AUTO}
            ),
            # note: wasn't able to narrow down the specific types that were failing
            # seems to be a little non-deterministic, but always fails with
            # cudaErrorInvalidValue invalid argument
            reason="libcudf json reader crashes on compressed non empty table_data",
        )
    )

    if isinstance(source, io.IOBase):
        source.seek(0)

    res = plc.io.json.read_json(
        plc.io.SourceInfo([source]),
        compression=compression_type,
        lines=lines,
    )

    # Adjustments to correct for the fact orient=records is lossy
    #  and doesn't
    # 1) preserve colnames when zero rows in table
    # 2) preserve struct nullability
    # 3) differentiate int64/uint64
    if len(pa_table) == 0:
        pa_table = pa.table([])

    new_fields = []
    for i in range(len(pa_table.schema)):
        curr_field = pa_table.schema.field(i)
        if curr_field.type == pa.uint64():
            try:
                curr_field = curr_field.with_type(pa.int64())
            except OverflowError:
                # There will be no confusion, values are too large
                # for int64 anyways
                pass
        new_fields.append(curr_field)

    pa_table = pa_table.cast(pa.schema(new_fields))

    # Convert non-nullable struct fields to nullable fields
    # since nullable=False cannot roundtrip through orient='records'
    # JSON format
    assert_table_and_meta_eq(pa_table, res, check_field_nullability=False)


def test_read_json_dtypes(table_data, source_or_sink):
    # Simple test for dtypes where we read in
    # all numeric data as floats
    _, pa_table = table_data
    source = make_json_source(
        source_or_sink,
        pa_table,
        lines=True,
    )

    dtypes = []
    new_fields = []
    for i in range(len(pa_table.schema)):
        field = pa_table.schema.field(i)
        child_types = []

        def get_child_types(typ):
            typ_child_types = []
            for i in range(typ.num_fields):
                curr_field = typ.field(i)
                typ_child_types.append(
                    (
                        curr_field.name,
                        curr_field.type,
                        get_child_types(curr_field.type),
                    )
                )
            return typ_child_types

        plc_type = plc.interop.from_arrow(field.type)
        if pa.types.is_integer(field.type) or pa.types.is_unsigned_integer(
            field.type
        ):
            plc_type = plc.interop.from_arrow(pa.float64())
            field = field.with_type(pa.float64())

        dtypes.append((field.name, plc_type, child_types))

        new_fields.append(field)

    new_schema = pa.schema(new_fields)

    res = plc.io.json.read_json(
        plc.io.SourceInfo([source]), dtypes=dtypes, lines=True
    )
    new_table = pa_table.cast(new_schema)

    # orient=records is lossy
    # and doesn't preserve column names when there's zero rows in the table
    if len(new_table) == 0:
        new_table = pa.table([])

    assert_table_and_meta_eq(new_table, res, check_field_nullability=False)


@pytest.mark.parametrize("chunk_size", [10, 15, 20])
def test_read_json_lines_byte_range(source_or_sink, chunk_size):
    source = source_or_sink
    if isinstance(source_or_sink, io.StringIO):
        pytest.skip("byte_range doesn't work on StringIO")

    json_str = "[1, 2, 3]\n[4, 5, 6]\n[7, 8, 9]\n"
    write_json_bytes(source, json_str)

    tbls_w_meta = []
    for chunk_start in range(0, len(json_str.encode("utf-8")), chunk_size):
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

    full_tbl_plc = plc.io.TableWithMetadata(
        plc.interop.from_arrow(full_tbl),
        tbls_w_meta[0].column_names(include_children=True),
    )
    assert_table_and_meta_eq(pa.Table.from_pandas(exp), full_tbl_plc)


@pytest.mark.parametrize("keep_quotes", [True, False])
def test_read_json_lines_keep_quotes(keep_quotes, source_or_sink):
    source = source_or_sink

    json_bytes = '["a", "b", "c"]\n'
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

    json_bytes = '{"a":1,"b":10}\n{"a":2,"b":11}\nabc\n{"a":3,"b":12}\n'
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


# TODO: Add tests for these!
# Tests were not added in the initial PR porting the JSON reader to pylibcudf
# to save time (and since there are no existing tests for these in Python cuDF)
# mixed_types_as_string = mixed_types_as_string,
# prune_columns = prune_columns,
