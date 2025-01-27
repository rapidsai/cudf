# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def data_col():
    pa_data_col = pa.array(
        ["a", "c", "A", "aa", None, "aaaaaaaaa", "AAAA", "ÁÁÁÁ"],
        type=pa.string(),
    )
    return pa_data_col, plc.interop.from_arrow(pa_data_col)


@pytest.fixture(scope="module", params=["a", "c", "A", "Á", "aa", "ÁÁÁ"])
def scalar_repl_target(request):
    pa_target = pa.scalar(request.param, type=pa.string())
    return request.param, plc.interop.from_arrow(pa_target)


@pytest.fixture(scope="module", params=["b", "B", "", "B́"])
def scalar_repl(request):
    pa_repl = pa.scalar(request.param, type=pa.string())
    return request.param, plc.interop.from_arrow(pa_repl)


@pytest.fixture(
    scope="module",
    params=[
        ["a", "c", "A", "ÁÁÁÁ"],
    ],
)
def col_repl_target(request):
    pa_target = pa.array(request.param, type=pa.string())
    return (pa_target, plc.interop.from_arrow(pa_target))


@pytest.fixture(
    scope="module",
    params=[
        [
            "",
            "z",
            "XX",
            "blahblah",
        ]
    ],
)
def col_repl(request):
    pa_repl = pa.array(request.param, type=pa.string())
    return (pa_repl, plc.interop.from_arrow(pa_repl))


@pytest.mark.parametrize("maxrepl", [-1, 1, 2, 10])
def test_replace(data_col, scalar_repl_target, scalar_repl, maxrepl):
    pa_data_col, plc_data_col = data_col
    pa_target, plc_target = scalar_repl_target
    pa_repl, plc_repl = scalar_repl
    got = plc.strings.replace.replace(
        plc_data_col, plc_target, plc_repl, maxrepl
    )

    expected = pc.replace_substring(
        pa_data_col,
        pattern=pa_target,
        replacement=pa_repl,
        max_replacements=maxrepl,
    )

    assert_column_eq(expected, got)


@pytest.mark.parametrize("startstop", [(0, -1), (0, 0), (1, 3)])
def test_replace_slice(data_col, scalar_repl, startstop):
    pa_data_col, plc_data_col = data_col
    pa_repl, plc_repl = scalar_repl
    start, stop = startstop
    got = plc.strings.replace.replace_slice(
        plc_data_col, plc_repl, start, stop
    )

    if stop == -1:
        # pyarrow doesn't support -1 as stop, so just set to really big number

        # TODO: once libcudf's count_characters() is migrated, we can call
        # count_characters on the input, take the max and set stop to that
        stop = 1000

    expected = pc.utf8_replace_slice(pa_data_col, start, stop, pa_repl)

    assert_column_eq(expected, got)


def test_replace_col(data_col, col_repl_target, col_repl):
    pa_data_col, plc_data_col = data_col
    pa_target, plc_target = col_repl_target
    pa_repl, plc_repl = col_repl
    got = plc.strings.replace.replace_multiple(
        plc_data_col, plc_target, plc_repl
    )

    # There's nothing in pyarrow that does string replace with columns
    # for targets/repls, so let's implement our own in python

    def replace_list(elem, targets, repls):
        for target, repl in zip(targets, repls):
            res = elem.replace(target, repl)
            if res != elem:
                return res

    targets = pa_target.to_pylist()
    repls = pa_repl.to_pylist()

    expected = pa.array(
        [
            replace_list(elem, targets, repls) if elem is not None else None
            for elem in pa_data_col.to_pylist()
        ],
        type=pa.string(),
    )

    assert_column_eq(expected, got)
