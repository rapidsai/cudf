import operator
import pytest

from numba import types
from numba.cuda import compile_ptx

from cudf import NA
from cudf.core.udf.typing import MaskedType

arith_ops = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
)

comparison_ops = (
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt
)

unary_ops = (
    operator.not_,
    operator.truth
)

ops = arith_ops + comparison_ops

number_types = (
    types.float32,
    types.float64,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
)

QUICK = False

if QUICK:
    arith_ops = (operator.add, operator.truediv, operator.pow)
    number_types = (types.int32, types.float32)


number_ids = tuple(str(t) for t in number_types)


@pytest.mark.parametrize('op', unary_ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
def test_compile_masked_unary(op, ty):

    def func(x):
        return op(x)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)


@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
@pytest.mark.parametrize('constant', [1, 1.5])
def test_compile_arith_masked_vs_constant(op, ty, constant):

    def func(x):
        return op(x, constant)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)

    # Check that the masked typing matches that of the unmasked typing
    um_ptx, um_resty = compile_ptx(func, (ty,), cc=cc, device=True)
    assert resty.value_type == um_resty


@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
@pytest.mark.parametrize('constant', [1, 1.5])
def test_compile_arith_constant_vs_masked(op, ty, constant):

    def func(x):
        return op(constant, x)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)


@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
def test_compile_arith_masked_vs_na(op, ty):

    def func(x):
        return op(x, NA)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)


@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
def test_compile_arith_na_vs_masked(op, ty):

    def func(x):
        return op(x, NA)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)


@pytest.mark.parametrize('op', ops)
@pytest.mark.parametrize('ty1', number_types, ids=number_ids)
@pytest.mark.parametrize('ty2', number_types, ids=number_ids)
@pytest.mark.parametrize('masked', ((False, True), (True, False),
                                    (True, True)),
                         ids=('um', 'mu', 'mm'))
def test_compile_arith_masked_ops(op, ty1, ty2, masked):

    def func(x, y):
        return op(x, y)

    cc = (7, 5)

    if masked[0]:
        ty1 = MaskedType(ty1)
    if masked[1]:
        ty2 = MaskedType(ty2)

    ptx, resty = compile_ptx(func, (ty1, ty2), cc=cc, device=True)
