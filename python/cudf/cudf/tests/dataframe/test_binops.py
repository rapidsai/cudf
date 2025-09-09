# Copyright (c) 2025, NVIDIA CORPORATION.
import operator

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "expected",
    [
        pd.RangeIndex(1, 2, name="a"),
        pd.Index([1], dtype=np.int8, name="a"),
        pd.MultiIndex.from_arrays([[1]], names=["a"]),
    ],
)
@pytest.mark.parametrize("binop", [lambda df: df == df, lambda df: df - 1])
def test_dataframe_binop_preserves_column_metadata(expected, binop):
    df = cudf.DataFrame([1], columns=expected)
    result = binop(df).columns
    pd.testing.assert_index_equal(result, expected, exact=True)


def test_dataframe_series_dot():
    pser = pd.Series(range(2))
    gser = cudf.from_pandas(pser)

    expected = pser @ pser
    actual = gser @ gser

    assert_eq(expected, actual)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"))
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pser], {}),
        rfunc_args_and_kwargs=([gser], {}),
    )

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pdf], {}),
        rfunc_args_and_kwargs=([gdf], {}),
    )

    pser = pd.Series(range(2), index=["a", "k"])
    gser = cudf.from_pandas(pser)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"), index=["a", "k"])
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    actual = gdf @ [2, 3]
    expected = pdf @ [2, 3]

    assert_eq(expected, actual)

    actual = pser @ [12, 13]
    expected = gser @ [12, 13]

    assert_eq(expected, actual)


def test_dataframe_binop_with_datetime_index():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(
        rng.random(2),
        index=pd.Index(
            [
                "2000-01-04",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
    )
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


def test_dataframe_binop_and_where():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.random(size=(2, 2)), columns=pd.Index([True, False]))
    gdf = cudf.from_pandas(df)

    expected = df > 1
    got = gdf > 1

    assert_eq(expected, got)

    expected = df[df > 1]
    got = gdf[gdf > 1]

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_string_types():
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame(rng.random(size=(3, 3)), columns=pd.Index([0, 1, 2]))
    df2 = pd.DataFrame(
        rng.random(size=(6, 6)),
        columns=pd.Index([0, 1, 2, "VhDoHxRaqt", "X0NNHBIPfA", "5FbhPtS0D1"]),
    )
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    expected = df2 + df1
    got = gdf2 + gdf1

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_date_types():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(rng.random(size=3), index=[0, 1, 2])
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"a": [10, 11, 12]}, index=["a", "b", "z"]),
        pd.DataFrame({"z": ["a"]}),
        pd.DataFrame({"a": [], "b": []}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": ["a", "a", "c", "z", "A"], "z": [1, 2, 3, 4, 5]}),
    ],
)
def test_dataframe_error_equality(df1, df2, comparison_op):
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    assert_exceptions_equal(
        comparison_op, comparison_op, ([df1, df2],), ([gdf1, gdf2],)
    )


@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_arithmetic_binops_df(arithmetic_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = arithmetic_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=arithmetic_op,
            rfunc=arithmetic_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = arithmetic_op(gdf, other)
        assert_eq(d, g)


@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_comparison_binops_df(comparison_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = comparison_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = comparison_op(gdf, other)
        assert_eq(d, g)


@pytest.mark.parametrize(
    "other",
    [
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
    ],
)
def test_comparison_binops_df_reindexing(request, comparison_op, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = comparison_op(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        request.applymarker(
            pytest.mark.xfail(
                condition=pdf.columns.difference(other.index).size > 0,
                reason="""
                Currently we will not match pandas for equality/inequality
                operators when there are columns that exist in a Series but not
                the DataFrame because pandas returns True/False values whereas
                we return NA. However, this reindexing is deprecated in pandas
                so we opt not to add support. This test should start passing
                once pandas removes the deprecated behavior in 2.0.  When that
                happens, this test can be merged with the two tests above into
                a single test with common parameters.
                """,
            )
        )

        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = comparison_op(gdf, other)
        assert_eq(d, g)


def test_binops_df_invalid():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    with pytest.raises(TypeError):
        gdf + np.array([1, 2])


def test_bitwise_binops_df(bitwise_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    d = bitwise_op(pdf, pdf + 1)
    g = bitwise_op(gdf, gdf + 1)
    assert_eq(d, g)


def test_binops_series(binary_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    pdf = pdf + 1.0
    gdf = gdf + 1.0
    d = binary_op(pdf.x, pdf.y)
    g = binary_op(gdf.x, gdf.y)
    assert_eq(d, g)


def test_bitwise_binops_series(bitwise_op):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    d = bitwise_op(pdf.x, pdf.y + 1)
    g = bitwise_op(gdf.x, gdf.y + 1)
    assert_eq(d, g)


@pytest.mark.parametrize("unaryop", [operator.neg, operator.inv, operator.abs])
@pytest.mark.parametrize(
    "col_name,assign_col_name", [(None, False), (None, True), ("abc", True)]
)
def test_unaryops_df(unaryop, col_name, assign_col_name):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    if assign_col_name:
        pdf.columns.name = col_name
    gdf = cudf.from_pandas(pdf)
    d = unaryop(pdf - 5)
    g = unaryop(gdf - 5)
    assert_eq(d, g)


def test_df_abs():
    rng = np.random.default_rng(seed=0)
    disturbance = pd.Series(rng.random(10))
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    pdf = pdf - 5 + disturbance
    d = np.abs(pdf)
    g = cudf.from_pandas(pdf).abs()
    assert_eq(d, g)


def test_scale_df():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    got = (gdf - 5).scale()
    expect = cudf.DataFrame(
        {"x": np.linspace(0.0, 1.0, 10), "y": np.linspace(0.0, 1.0, 10)}
    )
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3], index=["a", "b", "c"]),
        pd.Series([4, 2, 3], index=["a", "b", "d"]),
        pd.Series([4, 2], index=["a", "b"]),
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
    ],
)
@pytest.mark.parametrize("colnames", [["a", "b", "c"], [0, 1, 2]])
def test_df_sr_binop(psr, colnames, binary_op):
    data = [[3.0, 2.0, 5.0], [3.0, None, 5.0], [6.0, 7.0, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gsr = cudf.Series.from_pandas(psr).astype("float64")

    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas(nullable=True)

    psr = gsr.to_pandas(nullable=True)

    try:
        expect = binary_op(pdf, psr)
    except ValueError:
        with pytest.raises(ValueError):
            binary_op(gdf, gsr)
        with pytest.raises(ValueError):
            binary_op(psr, pdf)
        with pytest.raises(ValueError):
            binary_op(gsr, gdf)
    else:
        got = binary_op(gdf, gsr).to_pandas(nullable=True)
        assert_eq(expect, got, check_dtype=False, check_like=True)

        expect = binary_op(psr, pdf)
        got = binary_op(gsr, gdf).to_pandas(nullable=True)
        assert_eq(expect, got, check_dtype=False, check_like=True)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        # comparison ops will temporarily XFAIL
        # see PR  https://github.com/rapidsai/cudf/pull/7491
        pytest.param(operator.eq, marks=pytest.mark.xfail),
        pytest.param(operator.lt, marks=pytest.mark.xfail),
        pytest.param(operator.le, marks=pytest.mark.xfail),
        pytest.param(operator.gt, marks=pytest.mark.xfail),
        pytest.param(operator.ge, marks=pytest.mark.xfail),
        pytest.param(operator.ne, marks=pytest.mark.xfail),
    ],
)
def test_df_sr_binop_col_order(op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    gsr = cudf.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])
    psr = gsr.to_pandas()

    with expect_warning_if(
        op
        in {
            operator.eq,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
        },
        FutureWarning,
    ):
        expect = op(pdf, psr).astype("float")
    out = op(gdf, gsr).astype("float")
    got = out[expect.columns]

    assert_eq(expect, got)
