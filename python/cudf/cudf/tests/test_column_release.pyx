from libcpp.memory cimport unique_ptr

from cudf._lib.move cimport move
from cudf._lib.column cimport Column
from cudf.tests.utils import assert_eq
import cudf

import pytest
import cython


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [1, 2, None, 3],
        ['a', 'b', None, 'c'],
        ['a', 'b', 'c'],
        [None, None, None]
    ]
)
def test_release(data):
    cdef Column expect = cudf.core.column.as_column(data)

    cdef Column inp = expect.copy(deep=True)
    cdef Column got = Column.from_unique_ptr(
        move(inp.release())
    )

    assert_eq(expect, got)


def test_release_with_offset():
    cdef Column inp = cudf.core.column.as_column([1, 2, 3])[1:]

    with pytest.raises(TypeError):
        inp.release()
