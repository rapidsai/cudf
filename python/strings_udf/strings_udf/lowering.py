# Copyright (c) 2022, NVIDIA CORPORATION.

import operator
from functools import partial

from numba import cuda, types
from numba.core import cgutils
from numba.core.datamodel import default_manager
from numba.core.typing import signature as nb_signature
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

from strings_udf._lib.tables import get_character_flags_table_ptr
from strings_udf._typing import size_type, string_view, udf_string

character_flags_table_ptr = get_character_flags_table_ptr()

_STR_VIEW_PTR = types.CPointer(string_view)
_UDF_STRING_PTR = types.CPointer(udf_string)


# CUDA function declarations
# read-only (input is a string_view, output is a fixed with type)
_string_view_len = cuda.declare_device("len", size_type(_STR_VIEW_PTR))


def _declare_binary_func(lhs, rhs, out, name):
    # Declare a binary function
    return cuda.declare_device(
        name,
        out(lhs, rhs),
    )


def _declare_strip_func(name):
    return cuda.declare_device(
        name, size_type(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR)
    )


# A binary function of the form f(string, string) -> bool
_declare_bool_str_str_func = partial(
    _declare_binary_func, _STR_VIEW_PTR, _STR_VIEW_PTR, types.boolean
)

_declare_size_type_str_str_func = partial(
    _declare_binary_func, _STR_VIEW_PTR, _STR_VIEW_PTR, size_type
)

_string_view_contains = _declare_bool_str_str_func("contains")
_string_view_eq = _declare_bool_str_str_func("eq")
_string_view_ne = _declare_bool_str_str_func("ne")
_string_view_ge = _declare_bool_str_str_func("ge")
_string_view_le = _declare_bool_str_str_func("le")
_string_view_gt = _declare_bool_str_str_func("gt")
_string_view_lt = _declare_bool_str_str_func("lt")
_string_view_startswith = _declare_bool_str_str_func("startswith")
_string_view_endswith = _declare_bool_str_str_func("endswith")
_string_view_find = _declare_size_type_str_str_func("find")
_string_view_rfind = _declare_size_type_str_str_func("rfind")
_string_view_contains = _declare_bool_str_str_func("contains")
_string_view_strip = _declare_strip_func("strip")
_string_view_lstrip = _declare_strip_func("lstrip")
_string_view_rstrip = _declare_strip_func("rstrip")


# A binary function of the form f(string, int) -> bool
_declare_bool_str_int_func = partial(
    _declare_binary_func, _STR_VIEW_PTR, types.int64, types.boolean
)


_string_view_isdigit = _declare_bool_str_int_func("pyisdigit")
_string_view_isalnum = _declare_bool_str_int_func("pyisalnum")
_string_view_isalpha = _declare_bool_str_int_func("pyisalpha")
_string_view_isdecimal = _declare_bool_str_int_func("pyisdecimal")
_string_view_isnumeric = _declare_bool_str_int_func("pyisnumeric")
_string_view_isspace = _declare_bool_str_int_func("pyisspace")
_string_view_isupper = _declare_bool_str_int_func("pyisupper")
_string_view_islower = _declare_bool_str_int_func("pyislower")
_string_view_istitle = _declare_bool_str_int_func("pyistitle")


_string_view_count = cuda.declare_device(
    "pycount",
    size_type(_STR_VIEW_PTR, _STR_VIEW_PTR),
)


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


@cuda_lowering_registry.lower_cast(string_view, udf_string)
def cast_string_view_to_udf_string(context, builder, fromty, toty, val):
    sv_ptr = builder.alloca(default_manager[fromty].get_value_type())
    udf_str_ptr = builder.alloca(default_manager[toty].get_value_type())
    builder.store(val, sv_ptr)
    _ = context.compile_internal(
        builder,
        call_create_udf_string_from_string_view,
        nb_signature(types.void, _STR_VIEW_PTR, types.CPointer(udf_string)),
        (sv_ptr, udf_str_ptr),
    )
    result = cgutils.create_struct_proxy(udf_string)(
        context, builder, value=builder.load(udf_str_ptr)
    )

    return result._getvalue()


# utilities
_create_udf_string_from_string_view = cuda.declare_device(
    "udf_string_from_string_view",
    types.void(types.CPointer(string_view), types.CPointer(udf_string)),
)


def call_create_udf_string_from_string_view(sv, udf_str):
    _create_udf_string_from_string_view(sv, udf_str)


# String function implementations
def call_len_string_view(st):
    return _string_view_len(st)


@cuda_lower(len, string_view)
def len_impl(context, builder, sig, args):
    sv_ptr = builder.alloca(args[0].type)
    builder.store(args[0], sv_ptr)
    result = context.compile_internal(
        builder,
        call_len_string_view,
        nb_signature(size_type, _STR_VIEW_PTR),
        (sv_ptr,),
    )

    return result


def create_binary_string_func(binary_func, retty):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement a binary function
    of two strings.
    """

    def deco(cuda_func):
        @cuda_lower(binary_func, string_view, string_view)
        def binary_func_impl(context, builder, sig, args):
            lhs_ptr = builder.alloca(args[0].type)
            rhs_ptr = builder.alloca(args[1].type)
            builder.store(args[0], lhs_ptr)
            builder.store(args[1], rhs_ptr)

            # these conditional statements should compile out
            if retty != udf_string:
                # binary function of two strings yielding a fixed-width type
                # example: str.startswith(other) -> bool
                # shim functions can return the value through nb_retval
                result = context.compile_internal(
                    builder,
                    cuda_func,
                    nb_signature(retty, _STR_VIEW_PTR, _STR_VIEW_PTR),
                    (lhs_ptr, rhs_ptr),
                )
                return result
            else:
                # binary function of two strings yielding a new string
                # example: str.strip(other) -> str
                # shim functions can not return a struct due to C linkage
                # so we create a new udf_string and pass a pointer to it
                # for the shim function to write the output to. The return
                # value of compile_internal is therefore discarded (although
                # this may change in the future if we need to return error
                # codes, for instance).
                udf_str_ptr = builder.alloca(
                    default_manager[udf_string].get_value_type()
                )
                _ = context.compile_internal(
                    builder,
                    cuda_func,
                    size_type(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR),
                    (udf_str_ptr, lhs_ptr, rhs_ptr),
                )
                result = cgutils.create_struct_proxy(udf_string)(
                    context, builder, value=builder.load(udf_str_ptr)
                )
                return result._getvalue()

        return binary_func_impl

    return deco


@create_binary_string_func(operator.contains, types.boolean)
def contains_impl(st, substr):
    return _string_view_contains(st, substr)


@create_binary_string_func(operator.eq, types.boolean)
def eq_impl(st, rhs):
    return _string_view_eq(st, rhs)


@create_binary_string_func(operator.ne, types.boolean)
def ne_impl(st, rhs):
    return _string_view_ne(st, rhs)


@create_binary_string_func(operator.ge, types.boolean)
def ge_impl(st, rhs):
    return _string_view_ge(st, rhs)


@create_binary_string_func(operator.le, types.boolean)
def le_impl(st, rhs):
    return _string_view_le(st, rhs)


@create_binary_string_func(operator.gt, types.boolean)
def gt_impl(st, rhs):
    return _string_view_gt(st, rhs)


@create_binary_string_func(operator.lt, types.boolean)
def lt_impl(st, rhs):
    return _string_view_lt(st, rhs)


@create_binary_string_func("StringView.strip", udf_string)
def strip_impl(result, to_strip, strip_char):
    return _string_view_strip(result, to_strip, strip_char)


@create_binary_string_func("StringView.lstrip", udf_string)
def lstrip_impl(result, to_strip, strip_char):
    return _string_view_lstrip(result, to_strip, strip_char)


@create_binary_string_func("StringView.rstrip", udf_string)
def rstrip_impl(result, to_strip, strip_char):
    return _string_view_rstrip(result, to_strip, strip_char)


@create_binary_string_func("StringView.startswith", types.boolean)
def startswith_impl(sv, substr):
    return _string_view_startswith(sv, substr)


@create_binary_string_func("StringView.endswith", types.boolean)
def endswith_impl(sv, substr):
    return _string_view_endswith(sv, substr)


@create_binary_string_func("StringView.count", size_type)
def count_impl(st, substr):
    return _string_view_count(st, substr)


@create_binary_string_func("StringView.find", size_type)
def find_impl(sv, substr):
    return _string_view_find(sv, substr)


@create_binary_string_func("StringView.rfind", size_type)
def rfind_impl(sv, substr):
    return _string_view_rfind(sv, substr)


def create_unary_identifier_func(id_func):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement a unary function
    of a string.
    """

    def deco(cuda_func):
        @cuda_lower(id_func, string_view)
        def id_func_impl(context, builder, sig, args):
            str_ptr = builder.alloca(args[0].type)
            builder.store(args[0], str_ptr)

            # Lookup table required for conversion functions
            # must be resolved at runtime after context initialization,
            # therefore cannot be a global variable
            tbl_ptr = context.get_constant(
                types.int64, character_flags_table_ptr
            )
            result = context.compile_internal(
                builder,
                cuda_func,
                nb_signature(types.boolean, _STR_VIEW_PTR, types.int64),
                (str_ptr, tbl_ptr),
            )

            return result

        return id_func_impl

    return deco


@create_unary_identifier_func("StringView.isdigit")
def isdigit_impl(st, tbl):
    return _string_view_isdigit(st, tbl)


@create_unary_identifier_func("StringView.isalnum")
def isalnum_impl(st, tbl):
    return _string_view_isalnum(st, tbl)


@create_unary_identifier_func("StringView.isalpha")
def isalpha_impl(st, tbl):
    return _string_view_isalpha(st, tbl)


@create_unary_identifier_func("StringView.isnumeric")
def isnumeric_impl(st, tbl):
    return _string_view_isnumeric(st, tbl)


@create_unary_identifier_func("StringView.isdecimal")
def isdecimal_impl(st, tbl):
    return _string_view_isdecimal(st, tbl)


@create_unary_identifier_func("StringView.isspace")
def isspace_impl(st, tbl):
    return _string_view_isspace(st, tbl)


@create_unary_identifier_func("StringView.isupper")
def isupper_impl(st, tbl):
    return _string_view_isupper(st, tbl)


@create_unary_identifier_func("StringView.islower")
def islower_impl(st, tbl):
    return _string_view_islower(st, tbl)


@create_unary_identifier_func("StringView.istitle")
def istitle_impl(st, tbl):
    return _string_view_istitle(st, tbl)
