# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc

# Map pylibcudf interpolation options to pyarrow options
interp_mapping = {
    plc.types.Interpolation.LINEAR: "linear",
    plc.types.Interpolation.LOWER: "lower",
    plc.types.Interpolation.HIGHER: "higher",
    plc.types.Interpolation.MIDPOINT: "midpoint",
    plc.types.Interpolation.NEAREST: "nearest",
}


@pytest.fixture(scope="module", params=[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
def col_data(request, numeric_pa_type):
    pa_array = pa.array(request.param, type=numeric_pa_type)
    return pa_array, plc.interop.from_arrow(pa_array)


@pytest.fixture(
    scope="module",
    params=[
        {
            "arrays": [[1, 2, 3, 5, 4], [5.0, 6.0, 8.0, 7.0, 9.0]],
            "schema": pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.int64()),
                ]
            ),
        },
        {
            "arrays": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 2.0, 2.2, 2.3, 2.4, None, None, 3.5, 4.5, 5.5],
            ],
            "schema": pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.float64()),
                ]
            ),
        },
    ],
)
def plc_tbl_data(request):
    return plc.interop.from_arrow(pa.Table.from_arrays(**request.param))


@pytest.mark.parametrize("q", [[], [0], [0.5], [0.1, 0.5, 0.7, 0.9]])
@pytest.mark.parametrize("exact", [True, False])
def test_quantile(col_data, interp_opt, q, exact):
    pa_col_data, plc_col_data = col_data
    ordered_indices = plc.interop.from_arrow(
        pc.cast(pc.sort_indices(pa_col_data), pa.int32())
    )
    res = plc.quantiles.quantile(
        plc_col_data, q, interp_opt, ordered_indices, exact
    )

    pa_interp_opt = interp_mapping[interp_opt]

    if exact:
        pa_col_data = pc.cast(pa_col_data, pa.float64())

    if len(q) > 0:
        # pyarrow quantile doesn't support empty q
        exp = pc.quantile(pa_col_data, q=q, interpolation=pa_interp_opt)
    else:
        exp = pa.array([], type=pa.float64())

    if not exact:
        exp = pc.cast(exp, pa_col_data.type, safe=False)

    assert_column_eq(exp, res)


def _pyarrow_quantiles(
    pa_tbl_data,
    q,
    interp_opt=plc.types.Interpolation.NEAREST,
    sorted_opt=plc.types.Sorted.NO,
    column_order=None,
    null_precedence=None,
):
    """
    The pyarrow equivalent of plc.quantiles.quantiles

    Takes the same arguments (except input should be a pyarrow table instead of
    of a pylibcudf table)

    NOTE: This function doesn't support having different null precedences because of
    a lack of support in pyarrow.
    """
    if len(q) > 0:
        # pyarrow quantile doesn't support empty q
        pa_interp_opt = interp_mapping[interp_opt]

        if sorted_opt == plc.types.Sorted.NO:
            order_mapper = {
                plc.types.Order.ASCENDING: "ascending",
                plc.types.Order.DESCENDING: "descending",
            }
            if null_precedence is None:
                null_precedence = [plc.types.NullOrder.BEFORE] * len(
                    pa_tbl_data.columns
                )
            if column_order is None:
                column_order = [plc.types.Order.ASCENDING] * len(
                    pa_tbl_data.columns
                )

            if not all(
                [
                    null_prec == null_precedence[0]
                    for null_prec in null_precedence
                ]
            ):
                raise NotImplementedError(
                    "Having varying null precendences is not implemented!"
                )

            pa_tbl_data = pa_tbl_data.sort_by(
                [
                    (name, order_mapper[order])
                    for name, order in zip(
                        pa_tbl_data.column_names, column_order
                    )
                ],
                null_placement="at_start"
                if null_precedence[0] == plc.types.NullOrder.BEFORE
                else "at_end",
            )
        row_idxs = pc.quantile(
            np.arange(0, len(pa_tbl_data)), q=q, interpolation=pa_interp_opt
        )
        exp = pa_tbl_data.take(row_idxs)
    else:
        exp = pa.Table.from_arrays(
            [[] for _ in range(len(pa_tbl_data.schema))],
            schema=pa_tbl_data.schema,
        )
    return exp


@pytest.mark.parametrize(
    "q", [[], [0.1], [0.2], [0.3], [0.4], [0.5], [0.1, 0.5, 0.7, 0.9]]
)
@pytest.mark.parametrize(
    "column_order", [[plc.types.Order.ASCENDING, plc.types.Order.ASCENDING]]
)
@pytest.mark.parametrize(
    "null_precedence",
    [
        [plc.types.NullOrder.BEFORE, plc.types.NullOrder.BEFORE],
        [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
    ],
)
def test_quantiles(
    plc_tbl_data, interp_opt, q, sorted_opt, column_order, null_precedence
):
    if interp_opt in {
        plc.types.Interpolation.LINEAR,
        plc.types.Interpolation.MIDPOINT,
    }:
        pytest.skip(
            "interp cannot be an arithmetic interpolation strategy for quantiles"
        )

    pa_tbl_data = plc.interop.to_arrow(plc_tbl_data, ["a", "b"])

    exp = _pyarrow_quantiles(
        pa_tbl_data,
        q=q,
        interp_opt=interp_opt,
        sorted_opt=sorted_opt,
        column_order=column_order,
        null_precedence=null_precedence,
    )

    res = plc.quantiles.quantiles(
        plc_tbl_data, q, interp_opt, sorted_opt, column_order, null_precedence
    )

    assert_table_eq(exp, res)


@pytest.mark.parametrize(
    "invalid_interp",
    [plc.types.Interpolation.LINEAR, plc.types.Interpolation.MIDPOINT],
)
def test_quantiles_invalid_interp(plc_tbl_data, invalid_interp):
    with pytest.raises(ValueError):
        plc.quantiles.quantiles(
            plc_tbl_data, q=np.array([0.1]), interp=invalid_interp
        )


@pytest.mark.parametrize(
    "q",
    [[0.1], (0.1,), np.array([0.1])],
)
def test_quantile_q_array_like(col_data, q):
    pa_col_data, plc_col_data = col_data
    ordered_indices = plc.interop.from_arrow(
        pc.cast(pc.sort_indices(pa_col_data), pa.int32())
    )
    res = plc.quantiles.quantile(
        plc_col_data,
        q=q,
        ordered_indices=ordered_indices,
    )
    exp = pc.quantile(pa_col_data, q=q)
    assert_column_eq(exp, res)


@pytest.mark.parametrize(
    "q",
    [[0.1], (0.1,), np.array([0.1])],
)
def test_quantiles_q_array_like(plc_tbl_data, q):
    res = plc.quantiles.quantiles(plc_tbl_data, q=q)
    pa_tbl_data = plc.interop.to_arrow(plc_tbl_data, ["a", "b"])
    exp = _pyarrow_quantiles(pa_tbl_data, q=q)
    assert_table_eq(exp, res)
