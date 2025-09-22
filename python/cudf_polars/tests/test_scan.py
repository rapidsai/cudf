# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import decimal
from typing import TYPE_CHECKING

import pytest
from werkzeug import Response

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.versions import POLARS_VERSION_LT_131

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_httpserver import HTTPServer
    from werkzeug import Request


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
            "d": [
                decimal.Decimal("1.23"),
                None,
                decimal.Decimal("0.00"),
                None,
                decimal.Decimal("-5.67"),
                None,
            ],
        }
    )


@pytest.fixture(params=[None, ["a"], ["b", "a"]], ids=["all", "subset", "reordered"])
def columns(request, row_index):
    name, _ = row_index
    if name is not None and request.param is not None:
        return [name, *request.param]
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
def zlice(request):
    # For use in testing that we handle
    # polars slice pushdown correctly
    return request.param


@pytest.fixture(params=["csv", "ndjson", "parquet", "chunked_parquet"])
def format(request):
    return request.param


@pytest.fixture
def scan_fn(format):
    if format == "csv":
        return pl.scan_csv
    elif format == "ndjson":
        return pl.scan_ndjson
    else:
        return pl.scan_parquet


def test_scan(
    tmp_path, df, format, scan_fn, row_index, n_rows, columns, mask, zlice, request
):
    name, offset = row_index
    is_chunked = format == "chunked_parquet"
    if is_chunked:
        format = "parquet"
    make_partitioned_source(df, tmp_path / "file", format)
    request.applymarker(
        pytest.mark.xfail(
            condition=(n_rows is not None and scan_fn is pl.scan_ndjson),
            reason="libcudf does not support n_rows",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=(zlice is not None and scan_fn is pl.scan_ndjson),
            reason="slice pushdown not supported in the libcudf JSON reader",
        )
    )
    q = scan_fn(
        tmp_path / "file",
        row_index_name=name,
        row_index_offset=offset,
        n_rows=n_rows,
    )
    engine = pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": is_chunked})

    if zlice is not None:
        q = q.slice(*zlice)
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
    if isinstance(filename, list):
        source = [tmp_path / fn for fn in filename]
    else:
        source = tmp_path / filename
    q = pl.scan_csv(source, glob=glob, n_rows=n_rows, skip_rows=skiprows)

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
    make_partitioned_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", schema=schema)
    assert_gpu_result_equal(q)


def test_scan_ndjson_unsupported(df, tmp_path):
    make_partitioned_source(df, tmp_path / "file", "ndjson")
    q = pl.scan_ndjson(tmp_path / "file", ignore_errors=True)
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_parquet_nested_null_raises(tmp_path):
    df = pl.DataFrame({"a": pl.Series([None], dtype=pl.List(pl.Null))})

    df.write_parquet(tmp_path / "file.pq")

    q = pl.scan_parquet(tmp_path / "file.pq")

    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_parquet_only_row_index_raises(df, tmp_path):
    make_partitioned_source(df, tmp_path / "file", "parquet")
    q = pl.scan_parquet(tmp_path / "file", row_index_name="index").select("index")
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("n_rows", [None, 2])
def test_scan_include_file_path(request, tmp_path, format, scan_fn, df, n_rows):
    if n_rows is not None:
        df = df.head(n_rows)
    make_partitioned_source(df, tmp_path / "file", format)

    q = scan_fn(tmp_path / "file", include_file_paths="files", n_rows=n_rows)

    if format == "ndjson":
        assert_ir_translation_raises(q, NotImplementedError)
    elif format == "parquet":
        assert_gpu_result_equal(q, engine=NO_CHUNK_ENGINE)
    else:
        assert_gpu_result_equal(q)


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
    make_partitioned_source(df, path, "parquet")
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
@pytest.mark.parametrize(
    "filter", [None, pl.col("a") > 3], ids=["no_filters", "with_filters"]
)
def test_scan_parquet_chunked(
    large_df,
    chunk_read_limit,
    pass_read_limit,
    filter,
):
    if filter is None:
        q = large_df
    else:
        q = large_df.filter(filter)
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            raise_on_fail=True,
            parquet_options={
                "chunked": True,
                "chunk_read_limit": chunk_read_limit,
                "pass_read_limit": pass_read_limit,
            },
        ),
    )


def test_select_arbitrary_order_with_row_index_column(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(tmp_path / "df.parquet")
    q = pl.scan_parquet(tmp_path / "df.parquet", row_index_name="foo").select(
        [pl.col("a"), pl.col("foo")]
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "has_header,new_columns",
    [
        (True, None),
        (False, ["a", "b", "c"]),
    ],
)
def test_scan_csv_with_and_without_header(
    df, tmp_path, has_header, new_columns, row_index, columns, zlice
):
    path = tmp_path / "test.csv"
    make_partitioned_source(
        df, path, "csv", write_kwargs={"include_header": has_header}
    )

    name, offset = row_index

    q = pl.scan_csv(
        path,
        has_header=has_header,
        new_columns=new_columns,
        row_index_name=name,
        row_index_offset=offset,
    )

    if zlice is not None:
        q = q.slice(*zlice)
    if columns is not None:
        q = q.select(columns)

    assert_gpu_result_equal(q)


def test_scan_csv_without_header_and_new_column_names_raises(df, tmp_path):
    path = tmp_path / "test.csv"
    make_partitioned_source(df, path, "csv", write_kwargs={"include_header": False})
    q = pl.scan_csv(path, has_header=False)
    assert_ir_translation_raises(q, NotImplementedError)


def test_scan_with_row_index(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    df.write_csv(tmp_path / "test-0.csv")
    df.write_csv(tmp_path / "test-1.csv")

    q = pl.scan_csv(tmp_path / "test-*.csv", row_index_name="index", row_index_offset=0)
    assert_gpu_result_equal(q)


def test_scan_from_file_uri(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "out.parquet"
    df = pl.DataFrame({"a": 1})
    df.write_parquet(path)
    q = pl.scan_parquet(f"file://{path}")
    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize("chunked", [False, True])
def test_scan_parquet_remote(
    request, tmp_path: Path, df: pl.DataFrame, httpserver: HTTPServer, *, chunked: bool
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="remote IO not supported",
        )
    )
    path = tmp_path / "foo.parquet"
    df.write_parquet(path)
    bytes_ = path.read_bytes()
    size = len(bytes_)

    def head_handler(_: Request) -> Response:
        return Response(
            status=200,
            headers={
                "Content-Type": "parquet",
                "Accept-Ranges": "bytes",
                "Content-Length": size,
            },
        )

    def get_handler(req: Request) -> Response:
        # parse bytes=200-500 for example (the actual data)
        rng = req.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start, end = map(int, req.headers["Range"][6:].split("-"))
            mv = memoryview(bytes_)[start : end + 1]
            return Response(
                mv.tobytes(),
                status=206,
                headers={
                    "Content-Type": "parquet",
                    "Accept-Ranges": "bytes",
                    "Content-Length": len(mv),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                },
            )
        return Response(
            bytes_,
            status=200,
            headers={
                "Content-Type": "parquet",
                "Accept-Ranges": "bytes",
                "Content-Length": size,
            },
        )

    server_path = "/foo.parquet"
    httpserver.expect_request(server_path, method="HEAD").respond_with_handler(
        head_handler
    )
    httpserver.expect_request(server_path, method="GET").respond_with_handler(
        get_handler
    )

    q = pl.scan_parquet(httpserver.url_for(server_path))

    assert_gpu_result_equal(
        q, engine=pl.GPUEngine(raise_on_fail=True, parquet_options={"chunked": chunked})
    )


def test_scan_ndjson_remote(
    request, tmp_path: Path, df: pl.LazyFrame, httpserver: HTTPServer
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_131,
            reason="remote IO not supported",
        )
    )
    path = tmp_path / "foo.jsonl"
    df.write_ndjson(path)
    bytes_ = path.read_bytes()
    size = len(bytes_)

    def head_handler(_: Request) -> Response:
        return Response(
            status=200,
            headers={
                "Content-Type": "ndjson",
                "Content-Length": size,
            },
        )

    def get_handler(_: Request) -> Response:
        return Response(
            bytes_,
            status=200,
            headers={
                "Content-Type": "ndjson",
                "Content-Length": size,
            },
        )

    server_path = "/foo.jsonl"
    httpserver.expect_request(server_path, method="HEAD").respond_with_handler(
        head_handler
    )
    httpserver.expect_request(server_path, method="GET").respond_with_handler(
        get_handler
    )

    q = pl.scan_ndjson(httpserver.url_for(server_path))
    assert_gpu_result_equal(q)
