import operator

import pytest
from numba import cuda, types
from numba.cuda import compile_ptx

from cudf import NA
from cudf.core.udf.classes import Masked
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
    operator.gt,
)

unary_ops = (operator.truth,)

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


@pytest.mark.parametrize("op", unary_ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
def test_compile_masked_unary(op, ty):
    def func(x):
        return op(x)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)


@pytest.mark.parametrize("op", arith_ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
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
            raise RuntimeError("Expected r0 to be invalid")
        if not r1.valid:
            raise RuntimeError("Expected r1 to be valid")
        if u != r1.value:
            print("Values: ", u, r1.value)
            raise RuntimeError("u != r1.value")

    test_kernel[1, 1](1, 2)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
@pytest.mark.parametrize("constant", [1, 1.5])
def test_compile_arith_masked_vs_constant(op, ty, constant):
    def func(x):
        return op(x, constant)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)

    # Check that the masked typing matches that of the unmasked typing
    um_ptx, um_resty = compile_ptx(func, (ty,), cc=cc, device=True)
    assert resty.value_type == um_resty


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
@pytest.mark.parametrize("constant", [1, 1.5])
def test_compile_arith_constant_vs_masked(op, ty, constant):
    def func(x):
        return op(constant, x)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
def test_compile_arith_masked_vs_na(op, ty):
    def func(x):
        return op(x, NA)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)

    assert isinstance(resty, MaskedType)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
def test_compile_arith_na_vs_masked(op, ty):
    def func(x):
        return op(NA, x)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (MaskedType(ty),), cc=cc, device=True)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("ty1", number_types, ids=number_ids)
@pytest.mark.parametrize("ty2", number_types, ids=number_ids)
@pytest.mark.parametrize(
    "masked",
    ((False, True), (True, False), (True, True)),
    ids=("um", "mu", "mm"),
)
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


@pytest.mark.parametrize("fn", (func_x_is_na, func_na_is_x))
def test_is_na(fn):

    valid = Masked(1, True)
    invalid = Masked(1, False)

    device_fn = cuda.jit(device=True)(fn)

    @cuda.jit(debug=True)
    def test_kernel():
        valid_is_na = device_fn(valid)
        invalid_is_na = device_fn(invalid)

        if valid_is_na:
            raise RuntimeError("Valid masked value is NA and should not be")

        if not invalid_is_na:
            raise RuntimeError("Invalid masked value is not NA and should be")

    test_kernel[1, 1]()


def func_lt_na(x):
    return x < NA


def func_gt_na(x):
    return x > NA


def func_eq_na(x):
    return x == NA


def func_ne_na(x):
    return x != NA


def func_ge_na(x):
    return x >= NA


def func_le_na(x):
    return x <= NA


def func_na_lt(x):
    return x < NA


def func_na_gt(x):
    return x > NA


def func_na_eq(x):
    return x == NA


def func_na_ne(x):
    return x != NA


def func_na_ge(x):
    return x >= NA


def func_na_le(x):
    return x <= NA


na_comparison_funcs = (
    func_lt_na,
    func_gt_na,
    func_eq_na,
    func_ne_na,
    func_ge_na,
    func_le_na,
    func_na_lt,
    func_na_gt,
    func_na_eq,
    func_na_ne,
    func_na_ge,
    func_na_le,
)


@pytest.mark.parametrize("fn", na_comparison_funcs)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
def test_na_masked_comparisons(fn, ty):

    device_fn = cuda.jit(device=True)(fn)

    @cuda.jit(debug=True)
    def test_kernel():
        unmasked = ty(1)
        valid_masked = Masked(unmasked, True)
        invalid_masked = Masked(unmasked, False)

        valid_cmp_na = device_fn(valid_masked)
        invalid_cmp_na = device_fn(invalid_masked)

        if valid_cmp_na:
            raise RuntimeError("Valid masked value compared True with NA")

        if invalid_cmp_na:
            raise RuntimeError("Invalid masked value compared True with NA")

    test_kernel[1, 1]()


# xfail because scalars do not yet cast for a comparison to NA
@pytest.mark.xfail
@pytest.mark.parametrize("fn", na_comparison_funcs)
@pytest.mark.parametrize("ty", number_types, ids=number_ids)
def test_na_scalar_comparisons(fn, ty):

    device_fn = cuda.jit(device=True)(fn)

    @cuda.jit(debug=True)
    def test_kernel():
        unmasked = ty(1)

        unmasked_cmp_na = device_fn(unmasked)

        if unmasked_cmp_na:
            raise RuntimeError("Unmasked value compared True with NA")

    test_kernel[1, 1]()
