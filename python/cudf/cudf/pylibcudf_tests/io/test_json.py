# Copyright (c) 2024, NVIDIA CORPORATION.
import io

import pyarrow as pa
import pytest
from utils import sink_to_str

import cudf._lib.pylibcudf as plc


@pytest.mark.parametrize("rows_per_chunk", [8, 100])
@pytest.mark.parametrize("lines", [True, False])
def test_write_json_basic(table_data, source_or_sink, lines, rows_per_chunk):
    plc_table_w_meta, pa_table = table_data
    sink = source_or_sink

    kwargs = dict()
    if rows_per_chunk <= plc_table_w_meta.tbl.num_rows():
        kwargs["rows_per_chunk"] = rows_per_chunk

    plc.io.json.write_json(
        plc.io.SinkInfo([sink]), plc_table_w_meta, lines=lines, **kwargs
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
