import operator
import pytest

from numba import cuda, types
from numba.cuda import compile_ptx

from cudf import NA
from cudf.core.udf.typing import MaskedType
from cudf.core.udf.classes import Masked


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
    operator.truth,
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


@pytest.mark.parametrize('op', arith_ops)
@pytest.mark.parametrize('ty', number_types, ids=number_ids)
def test_execute_masked_binary(op, ty):

    @cuda.jit(device=True)
    def func(x, y):
        return op(x, y)

    @cuda.jit(debug=True)
    def test_kernel(x, y):
        # Reference result with unmasked value
        u = func(x, y)

        # Construct masked values to test with
        x0, y0 = Masked(x, False), Masked(y, False)
        x1, y1 = Masked(x, True), Masked(y, True)

        # Call with masked types
        r0 = func(x0, y0)
        r1 = func(x1, y1)

        # Check masks are as expected, and unmasked result matches masked
        # result
        if r0.valid:
            raise RuntimeError('Expected r0 to be invalid')
        if not r1.valid:
            raise RuntimeError('Expected r1 to be valid')
        if u != r1.value:
            print('Values: ', u, r1.value)
            raise RuntimeError('u != r1.value')

    test_kernel[1, 1](1, 2)


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
        return op(NA, x)

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


def func_x_is_na(x):
    return x is NA


def func_na_is_x(x):
    return NA is x


@pytest.mark.parametrize('fn', (func_x_is_na, func_na_is_x))
def test_is_na(fn):

    valid = Masked(1, True)
    invalid = Masked(1, False)

    device_fn = cuda.jit(device=True)(fn)

    @cuda.jit(debug=True)
    def test_kernel():
        valid_result = device_fn(valid)
        invalid_result = device_fn(invalid)

        if not valid_result:
            raise RuntimeError('Valid masked value is NA and should not be')

        if invalid_result:
            raise RuntimeError('Invalid masked value is not NA and should be')

    test_kernel[1, 1]()
