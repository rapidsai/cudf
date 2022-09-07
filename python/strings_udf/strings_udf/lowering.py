# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import cuda, types
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

from strings_udf._lib.tables import get_character_flags_table_ptr
from strings_udf._typing import size_type, string_view

character_flags_table_ptr = get_character_flags_table_ptr()


_string_view_len = cuda.declare_device(
    "len", size_type(types.CPointer(string_view))
)

_string_view_contains = cuda.declare_device(
    "contains",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)


_string_view_eq = cuda.declare_device(
    "eq",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)
_string_view_ne = cuda.declare_device(
    "ne",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_ge = cuda.declare_device(
    "ge",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_le = cuda.declare_device(
    "le",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)
_string_view_gt = cuda.declare_device(
    "gt",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_lt = cuda.declare_device(
    "lt",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_startswith = cuda.declare_device(
    "startswith",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_endswith = cuda.declare_device(
    "endswith",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_find = cuda.declare_device(
    "find",
    size_type(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_rfind = cuda.declare_device(
    "rfind",
    size_type(types.CPointer(string_view), types.CPointer(string_view)),
)

_string_view_isdigit = cuda.declare_device(
    "pyisdigit", types.boolean(types.CPointer(string_view), types.int64)
)


_string_view_isalnum = cuda.declare_device(
    "pyisalnum", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_isalpha = cuda.declare_device(
    "pyisalpha", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_isdecimal = cuda.declare_device(
    "pyisdecimal", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_isnumeric = cuda.declare_device(
    "pyisnumeric", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_isspace = cuda.declare_device(
    "pyisspace", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_isupper = cuda.declare_device(
    "pyisupper", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_islower = cuda.declare_device(
    "pyislower", types.boolean(types.CPointer(string_view), types.int64)
)

_string_view_count = cuda.declare_device(
    "pycount",
    size_type(types.CPointer(string_view), types.CPointer(string_view)),
)


# String function implementations
def call_len_string_view(st):
    return _string_view_len(st)


@cuda_lower(len, string_view)
def string_view_len_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    result = context.compile_internal(
        builder,
        call_len_string_view,
        nb_signature(size_type, types.CPointer(string_view)),
        (sv_ptr,),
    )

    return result


def call_string_view_contains(st, substr):
    return _string_view_contains(st, substr)


@cuda_lower(operator.contains, string_view, string_view)
def string_view_contains_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    substr_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)
    result = context.compile_internal(
        builder,
        call_string_view_contains,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_eq(st, rhs):
    return _string_view_eq(st, rhs)


@cuda_lower(operator.eq, string_view, string_view)
def string_view_eq_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_eq,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


def call_string_view_ne(st, rhs):
    return _string_view_ne(st, rhs)


@cuda_lower(operator.ne, string_view, string_view)
def string_view_ne_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_ne,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


def call_string_view_ge(st, rhs):
    return _string_view_ge(st, rhs)


@cuda_lower(operator.ge, string_view, string_view)
def string_view_ge_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_ge,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


def call_string_view_le(st, rhs):
    return _string_view_le(st, rhs)


@cuda_lower(operator.le, string_view, string_view)
def string_view_le_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_le,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


def call_string_view_gt(st, rhs):
    return _string_view_gt(st, rhs)


@cuda_lower(operator.gt, string_view, string_view)
def string_view_gt_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_gt,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


def call_string_view_lt(st, rhs):
    return _string_view_lt(st, rhs)


@cuda_lower(operator.lt, string_view, string_view)
def string_view_lt_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)

    builder.store(args[0], sv_ptr)
    builder.store(args[1], rhs_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_lt,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, rhs_ptr),
    )

    return result


# read-only functions
# We will provide only one overload for this set of functions, which will
# expect a string_view. When a literal is encountered, numba will promote it to
# a string_view whereas when a dstring is encountered, numba will convert it to
# a view via its native view() method.

# casts
@cuda_lowering_registry.lower_cast(types.StringLiteral, string_view)
def cast_string_literal_to_string_view(context, builder, fromty, toty, val):
    """
    Cast a literal to a string_view
    """
    # create an empty string_view
    sv = cgutils.create_struct_proxy(string_view)(context, builder)

    # set the empty strview data pointer to point to the literal value
    s = context.insert_const_string(builder.module, fromty.literal_value)
    sv.data = context.insert_addrspace_conv(
        builder, s, nvvm.ADDRSPACE_CONSTANT
    )
    sv.length = context.get_constant(size_type, len(fromty.literal_value))
    sv.bytes = context.get_constant(
        size_type, len(fromty.literal_value.encode("UTF-8"))
    )

    return sv._getvalue()


def call_string_view_startswith(sv, substr):
    return _string_view_startswith(sv, substr)


@cuda_lower("StringView.startswith", string_view, string_view)
def string_view_startswith_impl(context, builder, sig, args):
    sv_ptr, substr_ptr = builder.alloca(args[0].type), builder.alloca(
        args[1].type
    )

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_startswith,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_endswith(sv, substr):
    return _string_view_endswith(sv, substr)


@cuda_lower("StringView.endswith", string_view, string_view)
def string_view_endswith_impl(context, builder, sig, args):
    sv_ptr, substr_ptr = builder.alloca(args[0].type), builder.alloca(
        args[1].type
    )

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_endswith,
        nb_signature(
            types.boolean,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_count(st, substr):
    return _string_view_count(st, substr)


@cuda_lower("StringView.count", string_view, string_view)
def string_view_coount_impl(context, builder, sig, args):
    sv_ptr, substr_ptr = builder.alloca(args[0].type), builder.alloca(
        args[1].type
    )

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_count,
        nb_signature(
            size_type,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_find(sv, substr):
    return _string_view_find(sv, substr)


@cuda_lower("StringView.find", string_view, string_view)
def string_view_find_impl(context, builder, sig, args):
    sv_ptr, substr_ptr = builder.alloca(args[0].type), builder.alloca(
        args[1].type
    )

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_find,
        nb_signature(
            size_type,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_rfind(sv, substr):
    return _string_view_rfind(sv, substr)


@cuda_lower("StringView.rfind", string_view, string_view)
def string_view_rfind_impl(context, builder, sig, args):
    sv_ptr, substr_ptr = builder.alloca(args[0].type), builder.alloca(
        args[1].type
    )

    builder.store(args[0], sv_ptr)
    builder.store(args[1], substr_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_rfind,
        nb_signature(
            size_type,
            types.CPointer(string_view),
            types.CPointer(string_view),
        ),
        (sv_ptr, substr_ptr),
    )

    return result


def call_string_view_isdigit(st, tbl):
    return _string_view_isdigit(st, tbl)


@cuda_lower("StringView.isdigit", string_view)
def string_view_isdigit_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isdigit,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isalnum(st, tbl):
    return _string_view_isalnum(st, tbl)


@cuda_lower("StringView.isalnum", string_view)
def string_view_isalnum_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isalnum,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isalpha(st, tbl):
    return _string_view_isalpha(st, tbl)


@cuda_lower("StringView.isalpha", string_view)
def string_view_isalpha_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isalpha,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isnumeric(st, tbl):
    return _string_view_isnumeric(st, tbl)


@cuda_lower("StringView.isnumeric", string_view)
def string_view_isnumeric_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isnumeric,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isdecimal(st, tbl):
    return _string_view_isdecimal(st, tbl)


@cuda_lower("StringView.isdecimal", string_view)
def string_view_isdecimal_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isdecimal,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isspace(st, tbl):
    return _string_view_isspace(st, tbl)


@cuda_lower("StringView.isspace", string_view)
def string_view_isspace_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isspace,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_isupper(st, tbl):
    return _string_view_isupper(st, tbl)


@cuda_lower("StringView.isupper", string_view)
def string_view_isupper_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_isupper,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result


def call_string_view_islower(st, tbl):
    return _string_view_islower(st, tbl)


@cuda_lower("StringView.islower", string_view)
def string_view_islower_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    tbl_ptr = context.get_constant(types.int64, character_flags_table_ptr)

    result = context.compile_internal(
        builder,
        call_string_view_islower,
        nb_signature(types.boolean, types.CPointer(string_view), types.int64),
        (sv_ptr, tbl_ptr),
    )

    return result
