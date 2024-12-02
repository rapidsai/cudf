# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

NO_CHUNK_ENGINE = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": False})


@pytest.fixture(
    params=[(None, None), ("row-index", 0), ("index", 10)],
    ids=["no_row_index", "zero_offset_row_index", "offset_row_index"],
)
def row_index(request):
    return request.param


@pytest.fixture(
    params=[None, 3],
    ids=["all_rows", "some_rows"],
)
def n_rows(request):
    return request.param


@pytest.fixture(scope="module")
def df():
    # TODO: more dtypes
    return pl.DataFrame(
        {
            "a": [1, 2, 3, None, 4, 5],
            "b": ["ẅ", "x", "y", "z", "123", "abcd"],
            "c": [None, None, 4, 5, -1, 0],
        }
    )


@pytest.fixture(params=[None, ["a"], ["b", "a"]], ids=["all", "subset", "reordered"])
def columns(request, row_index):
    name, _ = row_index
    if name is not None and request.param is not None:
        return [*request.param, name]
    return request.param


@pytest.fixture(
    params=[None, pl.col("c").is_not_null()], ids=["no_mask", "c_is_not_null"]
)
def mask(request):
    return request.param


@pytest.fixture(
    params=[None, (1, 1)],
    ids=["no_slice", "slice_second"],
)
def slice(request):
    # For use in testing that we handle
    # polars slice pushdown correctly
    return request.param


def make_source(df, path, format):
    """
    Writes the passed polars df to a file of
    the desired format
    """
    if format == "csv":
        df.write_csv(path)
    elif format == "ndjson":
        df.write_ndjson(path)
    else:
        df.write_parquet(path)


@pytest.mark.parametrize(
    "format, scan_fn",
    [
        ("csv", pl.scan_csv),
        ("ndjson", pl.scan_ndjson),
        ("parquet", pl.scan_parquet),
        ("chunked_parquet", pl.scan_parquet),
    ],
)
def test_scan(
    tmp_path, df, format, scan_fn, row_index, n_rows, columns, mask, slice, request
):
    name, offset = row_index
    is_chunked = format == "chunked_parquet"
    if is_chunked:
        format = "parquet"
    make_source(df, tmp_path / "file", format)
    request.applymarker(
        pytest.mark.xfail(
            condition=(n_rows is not None and scan_fn is pl.scan_ndjson),
            reason="libcudf does not support n_rows",
        )
    )
    q = scan_fn(
        tmp_path / "file",
        row_index_name=name,
        row_index_offset=offset,
        n_rows=n_rows,
    )
    engine = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": is_chunked})

    if slice is not None:
        q = q.slice(*slice)
    if mask is not None:
        q = q.filter(mask)
    if columns is not None:
        q = q.select(*columns)
    assert_gpu_result_equal(q, engine=engine)


def test_negative_slice_pushdown_raises(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_parquet(tmp_path / "df.parquet")
    q = pl.scan_parquet(tmp_path / "df.parquet")
    # Take the last row
    q = q.slice(-1, 1)
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_unsupported_raises(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_ipc(tmp_path / "df.ipc")
    q = pl.scan_ipc(tmp_path / "df.ipc")
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_ndjson_nrows_notimplemented(tmp_path, df):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_ndjson(tmp_path / "df.jsonl")
    q = pl.scan_ndjson(tmp_path / "df.jsonl", n_rows=1)
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_row_index_projected_out(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_parquet(tmp_path / "df.pq")

    q = pl.scan_parquet(tmp_path / "df.pq").with_row_index().select(pl.col("a"))

    assert_gpu_result_equal(q, engine=NO_CHUNK_ENGINE)


def test_scan_csv_column_renames_projection_schema(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2\n3,4,5""")

    q = pl.scan_csv(
        tmp_path / "test.csv",
        with_column_names=lambda names: [f"{n}_suffix" for n in names],
        schema_overrides={
            "foo_suffix": pl.String(),
            "bar_suffix": pl.Int8(),
            "baz_suffix": pl.UInt16(),
        },
    )

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "filename,glob",
    [
        (["test1.csv", "test2.csv"], True),
        ("test*.csv", True),
        # Make sure we don't expand glob when
        # trying to read a file like test*.csv
        # when glob=False
        ("test*.csv", False),
    ],
)
@pytest.mark.parametrize(
    "nrows_skiprows",
    [
        (None, 0),
        (1, 1),
        (3, 0),
        (4, 2),
    ],
)
def test_scan_csv_multi(tmp_path, filename, glob, nrows_skiprows):
    n_rows, skiprows = nrows_skiprows
    with (tmp_path / "test1.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    with (tmp_path / "test2.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    with (tmp_path / "test*.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")
    os.chdir(tmp_path)
    q = pl.scan_csv(filename, glob=glob, n_rows=n_rows, skip_rows=skiprows)

    assert_gpu_result_equal(q)


def test_scan_csv_multi_differing_colnames(tmp_path):
    with (tmp_path / "test1.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2\n3,4,5""")
    with (tmp_path / "test2.csv").open("w") as f:
        f.write("""abc,def,ghi\n1,2\n3,4,5""")
    q = pl.scan_csv(
        [tmp_path / "test1.csv", tmp_path / "test2.csv"],
    )
    with pytest.raises(pl.exceptions.ComputeError):
        q.explain()


def test_scan_csv_skip_after_header_not_implemented(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", skip_rows_after_header=1)

    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_csv_null_values_per_column_not_implemented(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", null_values={"foo": "1", "baz": "5"})

    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_csv_comment_str_not_implemented(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n// 1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", comment_prefix="// ")

    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_csv_comment_char(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n# 1,2,3\n3,4,5""")

    q = pl.scan_csv(tmp_path / "test.csv", comment_prefix="#")

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("nulls", [None, "3", ["3", "5"]])
def test_scan_csv_null_values(tmp_path, nulls):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo,bar,baz\n1,2,3\n3,4,5\n5,,2""")

    q = pl.scan_csv(tmp_path / "test.csv", null_values=nulls)

    assert_gpu_result_equal(q)


def test_scan_csv_decimal_comma(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""foo|bar|baz\n1,23|2,34|3,56\n1""")

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", decimal_comma=True)

    assert_gpu_result_equal(q)


def test_scan_csv_skip_initial_empty_rows(tmp_path):
    with (tmp_path / "test.csv").open("w") as f:
        f.write("""\n\n\n\nfoo|bar|baz\n1|2|3\n1""")

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", skip_rows=1, has_header=False)

    assert_ir_translation_raises(q, NotImplementedError)

    q = pl.scan_csv(tmp_path / "test.csv", separator="|", skip_rows=1)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "schema",
    [
        # List of colnames (basicaly like names param in CSV)
        {"b": pl.String, "a": pl.Float32},
        {"a": pl.UInt64},
    ],
)
def test_scan_ndjson_schema(df, tmp_path, schema):
    make_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", schema=schema)
    assert_gpu_result_equal(q)


def test_scan_ndjson_unsupported(df, tmp_path):
    make_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", ignore_errors=True)
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_parquet_nested_null_raises(tmp_path):
    df = pl.DataFrame({"a": pl.Series([None], dtype=pl.List(pl.Null))})

    df.write_parquet(tmp_path / "file.pq")

    q = pl.scan_parquet(tmp_path / "file.pq")

    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_parquet_only_row_index_raises(df, tmp_path):
    make_source(df, tmp_path / "file", "parquet")
    q = pl.scan_parquet(tmp_path / "file", row_index_name="index").select("index")
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.fixture(
    scope="module", params=["no_slice", "skip_to_end", "skip_partial", "partial"]
)
def chunked_slice(request):
    return request.param


@pytest.fixture(scope="module")
def large_df(df, tmpdir_factory, chunked_slice):
    # Something big enough that we get more than a single chunk,
    # empirically determined
    df = pl.concat([df] * 1000)
    df = pl.concat([df] * 10)
    df = pl.concat([df] * 10)
    path = str(tmpdir_factory.mktemp("data") / "large.pq")
    make_source(df, path, "parquet")
    n_rows = len(df)
    q = pl.scan_parquet(path)
    if chunked_slice == "no_slice":
        return q
    elif chunked_slice == "skip_to_end":
        return q.slice(int(n_rows * 0.6), n_rows)
    elif chunked_slice == "skip_partial":
        return q.slice(int(n_rows * 0.6), int(n_rows * 0.2))
    else:
        return q.slice(0, int(n_rows * 0.6))


@pytest.mark.parametrize(
    "chunk_read_limit", [0, 1, 2, 4, 8, 16], ids=lambda x: f"chunk_{x}"
)
@pytest.mark.parametrize(
    "pass_read_limit", [0, 1, 2, 4, 8, 16], ids=lambda x: f"pass_{x}"
)
def test_scan_parquet_chunked(
    request, chunked_slice, large_df, chunk_read_limit, pass_read_limit
):
    assert_gpu_result_equal(
        large_df,
        engine=pl.GPUEngine(
            raise_on_fail=True,
            parquet_options={
                "chunked": True,
                "chunk_read_limit": chunk_read_limit,
                "pass_read_limit": pass_read_limit,
            },
        ),
    )


def test_scan_hf_url_raises():
    q = pl.scan_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    assert_ir_translation_raises(q, NotImplementedError)
