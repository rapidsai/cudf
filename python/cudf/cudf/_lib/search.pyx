# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from cudf._lib import pylibcudf


@acquire_spill_lock()
def search_sorted(
    list source, list values, side, ascending=True, na_position="last"
):
    """Find indices where elements should be inserted to maintain order

    Parameters
    ----------
    source : list of columns
        List of columns to search in
    values : List of columns
        List of value columns to search for
    side : str {'left', 'right'} optional
        If 'left', the index of the first suitable location is given.
        If 'right', return the last such index
    """
    # Note: We are ignoring index columns here
    column_order = [
        pylibcudf.types.Order.ASCENDING
        if ascending
        else pylibcudf.types.Order.DESCENDING
    ] * len(source)
    null_precedence = [
        pylibcudf.types.NullOrder.AFTER
        if na_position == "last"
        else pylibcudf.types.NullOrder.BEFORE
    ] * len(source)

    func = getattr(
        pylibcudf.search,
        "lower_bound" if side == "left" else "upper_bound",
    )
    return Column.from_pylibcudf(
        func(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in source]),
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in values]),
            column_order,
            null_precedence,
        )
    )


@acquire_spill_lock()
def contains(Column haystack, Column needles):
    """Check whether column contains multiple values

    Parameters
    ----------
    column : NumericalColumn
        Column to search in
    needles :
        A column of values to search for
    """
    return Column.from_pylibcudf(
        pylibcudf.search.contains(
            haystack.to_pylibcudf(mode="read"),
            needles.to_pylibcudf(mode="read"),
        )
    )
