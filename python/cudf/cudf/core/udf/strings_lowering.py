# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import partial

from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)

from cudf._lib.strings_udf import (
    get_character_cases_table_ptr,
    get_character_flags_table_ptr,
    get_special_case_mapping_table_ptr,
)
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import (
    NRT_decref,
    managed_udf_string,
    size_type,
    string_view,
    udf_string,
)

_STR_VIEW_PTR = types.CPointer(string_view)
_UDF_STRING_PTR = types.CPointer(udf_string)


# CUDA function declarations
# read-only (input is a string_view, output is a fixed with type)
_string_view_len = cuda.declare_device("len", size_type(_STR_VIEW_PTR))


_concat_string_view = cuda.declare_device(
    "concat", types.voidptr(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR)
)

_string_view_replace = cuda.declare_device(
    "replace",
    types.voidptr(
        _UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR
    ),
)


def _declare_binary_func(lhs, rhs, out, name):
    # Declare a binary function
    return cuda.declare_device(
        name,
        out(lhs, rhs),
    )


def _declare_strip_func(name):
    return cuda.declare_device(
        name, types.voidptr(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR)
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


def _declare_upper_or_lower(func):
    return cuda.declare_device(
        func,
        types.voidptr(
            _UDF_STRING_PTR,
            _STR_VIEW_PTR,
            types.uintp,
            types.uintp,
            types.uintp,
        ),
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
_string_view_upper = _declare_upper_or_lower("upper")
_string_view_lower = _declare_upper_or_lower("lower")


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
    sv.data = context.insert_string_const_addrspace(
        builder, fromty.literal_value
    )
    sv.length = context.get_constant(size_type, len(fromty.literal_value))
    sv.bytes = context.get_constant(
        size_type, len(fromty.literal_value.encode("UTF-8"))
    )

    return sv._getvalue()


@cuda_lowering_registry.lower_cast(string_view, managed_udf_string)
def cast_string_view_to_managed_udf_string(
    context, builder, fromty, toty, val
):
    sv_ptr = builder.alloca(
        context.data_model_manager[fromty].get_value_type()
    )

    managed_ptr = builder.alloca(
        context.data_model_manager[managed_udf_string].get_value_type()
    )
    udf_str_ptr = builder.gep(
        managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
    )

    builder.store(val, sv_ptr)
    meminfo = context.compile_internal(
        builder,
        call_create_udf_string_from_string_view,
        nb_signature(types.voidptr, _STR_VIEW_PTR, types.CPointer(udf_string)),
        (sv_ptr, udf_str_ptr),
    )
    managed = cgutils.create_struct_proxy(managed_udf_string)(
        context,
        builder,
        value=builder.load(managed_ptr),
    )
    managed.meminfo = meminfo

    return managed._getvalue()


@cuda_lowering_registry.lower_cast(managed_udf_string, string_view)
def cast_managed_udf_string_to_string_view(
    context, builder, fromty, toty, val
):
    sv_ptr = builder.alloca(context.data_model_manager[toty].get_value_type())

    managed_ptr = builder.alloca(
        context.data_model_manager[fromty].get_value_type()
    )
    builder.store(val, managed_ptr)

    udf_str_ptr = builder.gep(
        managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
    )

    context.compile_internal(
        builder,
        call_create_string_view_from_udf_string,
        nb_signature(types.void, _UDF_STRING_PTR, _STR_VIEW_PTR),
        (udf_str_ptr, sv_ptr),
    )
    result = cgutils.create_struct_proxy(string_view)(
        context, builder, value=builder.load(sv_ptr)
    )
    return result._getvalue()


# Utilities and Casts
_create_udf_string_from_string_view = cuda.declare_device(
    "udf_string_from_string_view",
    types.void(_STR_VIEW_PTR, _UDF_STRING_PTR),
)
_create_string_view_from_udf_string = cuda.declare_device(
    "string_view_from_udf_string",
    types.void(_UDF_STRING_PTR, _STR_VIEW_PTR),
)


def call_create_udf_string_from_string_view(sv, udf_str):
    _create_udf_string_from_string_view(sv, udf_str)


def call_create_string_view_from_udf_string(udf_str, sv):
    _create_string_view_from_udf_string(udf_str, sv)


# Return string setitem impl with an extra incref
@cuda_lower(
    operator.setitem,
    types.CPointer(managed_udf_string),
    types.Integer,
    types.Any,
)
def setitem_cpointer_managed_udf_string(context, builder, sig, args):
    base_ptr, idx, val = args
    elem_ptr = builder.gep(base_ptr, [idx])
    builder.store(val, elem_ptr)
    # Storing a Managed UDF String in a CPointer array effectively creates a
    # new reference; represent this by incrementing the refcount of the source
    # of the assignment
    context.nrt.incref(builder, managed_udf_string, val)


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


@cuda_lower(NRT_decref, managed_udf_string)
def decref_managed_udf_string(context, builder, sig, args):
    managed_ptr = args[0]
    managed = cgutils.create_struct_proxy(managed_udf_string)(
        context, builder, value=managed_ptr
    )
    fnty = ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.IntType(8))])
    fn = cgutils.get_or_insert_function(
        builder.module, fnty, "NRT_decref_managed_string"
    )
    builder.call(fn, (managed.meminfo,))

    return


def call_concat_string_view(result, lhs, rhs):
    return _concat_string_view(result, lhs, rhs)


@cuda_lower(operator.add, string_view, string_view)
def concat_impl(context, builder, sig, args):
    lhs_ptr = builder.alloca(args[0].type)
    rhs_ptr = builder.alloca(args[1].type)
    builder.store(args[0], lhs_ptr)
    builder.store(args[1], rhs_ptr)

    managed_ptr = builder.alloca(
        context.data_model_manager[managed_udf_string].get_value_type()
    )

    udf_str_ptr = builder.gep(
        managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
    )

    meminfo = context.compile_internal(
        builder,
        call_concat_string_view,
        types.voidptr(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR),
        (udf_str_ptr, lhs_ptr, rhs_ptr),
    )

    managed = cgutils.create_struct_proxy(managed_udf_string)(
        context,
        builder,
        value=builder.load(managed_ptr),
    )
    managed.meminfo = meminfo

    return managed._getvalue()


def call_string_view_replace(result, src, to_replace, replacement):
    return _string_view_replace(result, src, to_replace, replacement)


@cuda_lower("StringView.replace", string_view, string_view, string_view)
def replace_impl(context, builder, sig, args):
    src_ptr = builder.alloca(args[0].type)
    to_replace_ptr = builder.alloca(args[1].type)
    replacement_ptr = builder.alloca(args[2].type)

    builder.store(args[0], src_ptr)
    builder.store(args[1], to_replace_ptr)
    builder.store(args[2], replacement_ptr)

    managed_ptr = builder.alloca(
        context.data_model_manager[managed_udf_string].get_value_type()
    )
    udf_str_ptr = builder.gep(
        managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
    )

    meminfo = context.compile_internal(
        builder,
        call_string_view_replace,
        types.voidptr(
            _UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR
        ),
        (udf_str_ptr, src_ptr, to_replace_ptr, replacement_ptr),
    )
    managed = cgutils.create_struct_proxy(managed_udf_string)(
        context,
        builder,
        value=builder.load(managed_ptr),
    )
    managed.meminfo = meminfo
    return managed._getvalue()


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

                managed_ptr = builder.alloca(
                    context.data_model_manager[
                        managed_udf_string
                    ].get_value_type()
                )
                udf_str_ptr = builder.gep(
                    managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
                )

                meminfo = context.compile_internal(
                    builder,
                    cuda_func,
                    types.voidptr(
                        _UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR
                    ),
                    (udf_str_ptr, lhs_ptr, rhs_ptr),
                )
                managed = cgutils.create_struct_proxy(managed_udf_string)(
                    context,
                    builder,
                    value=builder.load(
                        managed_ptr
                    ),  # {i8*, {i8*, i32, i32}}* -> {i8*, {i8*, i32, i32}}
                )
                managed.meminfo = meminfo

                return managed._getvalue()

        # binary_func can be attribute-like: str.binary_func
        # or operator-like: binary_func(str, other)
        if isinstance(binary_func, str):
            binary_func_impl = cuda_lower(
                f"StringView.{binary_func}", string_view, string_view
            )(binary_func_impl)
            binary_func_impl = cuda_lower(
                f"ManagedUDFString.{binary_func}", string_view, string_view
            )(binary_func_impl)
        else:
            binary_func_impl = cuda_lower(
                binary_func, string_view, string_view
            )(binary_func_impl)

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


@create_binary_string_func("strip", udf_string)
def strip_impl(result, to_strip, strip_char):
    return _string_view_strip(result, to_strip, strip_char)


@create_binary_string_func("lstrip", udf_string)
def lstrip_impl(result, to_strip, strip_char):
    return _string_view_lstrip(result, to_strip, strip_char)


@create_binary_string_func("rstrip", udf_string)
def rstrip_impl(result, to_strip, strip_char):
    return _string_view_rstrip(result, to_strip, strip_char)


@create_binary_string_func("startswith", types.boolean)
def startswith_impl(sv, substr):
    return _string_view_startswith(sv, substr)


@create_binary_string_func("endswith", types.boolean)
def endswith_impl(sv, substr):
    return _string_view_endswith(sv, substr)


@create_binary_string_func("count", size_type)
def count_impl(st, substr):
    return _string_view_count(st, substr)


@create_binary_string_func("find", size_type)
def find_impl(sv, substr):
    return _string_view_find(sv, substr)


@create_binary_string_func("rfind", size_type)
def rfind_impl(sv, substr):
    return _string_view_rfind(sv, substr)


def create_unary_identifier_func(id_func):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement a unary function
    of a string.
    """

    def deco(cuda_func):
        @cuda_lower(f"StringView.{id_func}", string_view)
        @cuda_lower(f"UDFString.{id_func}", string_view)
        def id_func_impl(context, builder, sig, args):
            str_ptr = builder.alloca(args[0].type)
            builder.store(args[0], str_ptr)

            # Lookup table required for conversion functions
            # must be resolved at runtime after context initialization,
            # therefore cannot be a global variable
            tbl_ptr = context.get_constant(
                types.uintp, get_character_flags_table_ptr()
            )
            result = context.compile_internal(
                builder,
                cuda_func,
                nb_signature(types.boolean, _STR_VIEW_PTR, types.uintp),
                (str_ptr, tbl_ptr),
            )

            return result

        return id_func_impl

    return deco


def create_upper_or_lower(id_func):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement either the upper
    or lower attrs of a string view.
    """

    def deco(cuda_func):
        @cuda_lower(f"StringView.{id_func}", string_view)
        @cuda_lower(f"UDFString.{id_func}", string_view)
        def id_func_impl(context, builder, sig, args):
            str_ptr = builder.alloca(args[0].type)
            builder.store(args[0], str_ptr)

            # Lookup table required for conversion functions
            # must be resolved at runtime after context initialization,
            # therefore cannot be a global variable
            flags_tbl_ptr = context.get_constant(
                types.uintp, get_character_flags_table_ptr()
            )
            cases_tbl_ptr = context.get_constant(
                types.uintp, get_character_cases_table_ptr()
            )
            special_tbl_ptr = context.get_constant(
                types.uintp, get_special_case_mapping_table_ptr()
            )

            managed_ptr = builder.alloca(
                context.data_model_manager[managed_udf_string].get_value_type()
            )
            udf_str_ptr = builder.gep(
                managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)]
            )
            meminfo = context.compile_internal(
                builder,
                cuda_func,
                types.voidptr(
                    _UDF_STRING_PTR,
                    _STR_VIEW_PTR,
                    types.uintp,
                    types.uintp,
                    types.uintp,
                ),
                (
                    udf_str_ptr,
                    str_ptr,
                    flags_tbl_ptr,
                    cases_tbl_ptr,
                    special_tbl_ptr,
                ),
            )
            managed = cgutils.create_struct_proxy(managed_udf_string)(
                context,
                builder,
                value=builder.load(
                    managed_ptr
                ),  # {i8*, {i8*, i32, i32}}* -> {i8*, {i8*, i32, i32}}
            )
            managed.meminfo = meminfo
            return managed._getvalue()

        return id_func_impl

    return deco


@create_upper_or_lower("upper")
def upper_impl(result, st, flags, cases, special):
    return _string_view_upper(result, st, flags, cases, special)


@create_upper_or_lower("lower")
def lower_impl(result, st, flags, cases, special):
    return _string_view_lower(result, st, flags, cases, special)


@create_unary_identifier_func("isdigit")
def isdigit_impl(st, tbl):
    return _string_view_isdigit(st, tbl)


@create_unary_identifier_func("isalnum")
def isalnum_impl(st, tbl):
    return _string_view_isalnum(st, tbl)


@create_unary_identifier_func("isalpha")
def isalpha_impl(st, tbl):
    return _string_view_isalpha(st, tbl)


@create_unary_identifier_func("isnumeric")
def isnumeric_impl(st, tbl):
    return _string_view_isnumeric(st, tbl)


@create_unary_identifier_func("isdecimal")
def isdecimal_impl(st, tbl):
    return _string_view_isdecimal(st, tbl)


@create_unary_identifier_func("isspace")
def isspace_impl(st, tbl):
    return _string_view_isspace(st, tbl)


@create_unary_identifier_func("isupper")
def isupper_impl(st, tbl):
    return _string_view_isupper(st, tbl)


@create_unary_identifier_func("islower")
def islower_impl(st, tbl):
    return _string_view_islower(st, tbl)


@create_unary_identifier_func("istitle")
def istitle_impl(st, tbl):
    return _string_view_istitle(st, tbl)


@cuda_lower(len, MaskedType(string_view))
@cuda_lower(len, MaskedType(udf_string))
def masked_len_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    result = len_impl(
        context, builder, size_type(string_view), (masked_sv.value,)
    )
    ret.value = result
    ret.valid = masked_sv.valid

    return ret._getvalue()


def _masked_proxies(context, builder, maskedty, *args):
    return tuple(
        cgutils.create_struct_proxy(maskedty)(context, builder, value=arg)
        for arg in args
    )


@cuda_lower(
    "MaskedType.replace",
    MaskedType(string_view),
    MaskedType(string_view),
    MaskedType(string_view),
)
def masked_string_view_replace_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    src_masked, to_replace_masked, replacement_masked = _masked_proxies(
        context, builder, MaskedType(string_view), *args
    )
    result = replace_impl(
        context,
        builder,
        nb_signature(udf_string, string_view, string_view, string_view),
        (src_masked.value, to_replace_masked.value, replacement_masked.value),
    )

    ret.value = result
    ret.valid = builder.and_(
        builder.and_(src_masked.valid, to_replace_masked.valid),
        replacement_masked.valid,
    )

    return ret._getvalue()


def create_masked_binary_string_func(op, cuda_func, retty):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement a binary function
    of two masked strings.
    """

    def masked_binary_func_impl(context, builder, sig, args):
        ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)

        lhs_masked = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        rhs_masked = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[1]
        )

        result = cuda_func(
            context,
            builder,
            nb_signature(retty, string_view, string_view),
            (lhs_masked.value, rhs_masked.value),
        )

        ret.value = result
        ret.valid = builder.and_(lhs_masked.valid, rhs_masked.valid)

        return ret._getvalue()

    cuda_lower(op, MaskedType(string_view), MaskedType(string_view))(
        masked_binary_func_impl
    )


def create_masked_unary_identifier_func(op, cuda_func):
    """
    Provide a wrapper around numba's low-level extension API which
    produces the boilerplate needed to implement a unary function
    of a masked string.
    """

    def masked_unary_func_impl(context, builder, sig, args):
        ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        masked_str = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )

        result = cuda_func(
            context,
            builder,
            types.boolean(string_view, string_view),
            (masked_str.value,),
        )
        ret.value = result
        ret.valid = masked_str.valid
        return ret._getvalue()

    cuda_lower(op, MaskedType(string_view))(masked_unary_func_impl)


def create_masked_upper_or_lower(op, cuda_func):
    def upper_or_lower_impl(context, builder, sig, args):
        ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        masked_str = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )

        result = cuda_func(
            context,
            builder,
            udf_string(string_view),
            (masked_str.value,),
        )
        ret.value = result
        ret.valid = masked_str.valid
        return ret._getvalue()

    cuda_lower(op, MaskedType(string_view))(upper_or_lower_impl)


create_masked_binary_string_func("MaskedType.strip", strip_impl, udf_string)
create_masked_binary_string_func("MaskedType.lstrip", lstrip_impl, udf_string)
create_masked_binary_string_func("MaskedType.rstrip", rstrip_impl, udf_string)
create_masked_binary_string_func(
    "MaskedType.startswith",
    startswith_impl,
    types.boolean,
)
create_masked_binary_string_func(
    "MaskedType.endswith", endswith_impl, types.boolean
)
create_masked_binary_string_func("MaskedType.find", find_impl, size_type)
create_masked_binary_string_func("MaskedType.rfind", rfind_impl, size_type)
create_masked_binary_string_func("MaskedType.count", count_impl, size_type)
create_masked_binary_string_func(
    operator.contains, contains_impl, types.boolean
)


create_masked_unary_identifier_func("MaskedType.isalnum", isalnum_impl)
create_masked_unary_identifier_func("MaskedType.isalpha", isalpha_impl)
create_masked_unary_identifier_func("MaskedType.isdigit", isdigit_impl)
create_masked_unary_identifier_func("MaskedType.isupper", isupper_impl)
create_masked_unary_identifier_func("MaskedType.islower", islower_impl)
create_masked_unary_identifier_func("MaskedType.isspace", isspace_impl)
create_masked_unary_identifier_func("MaskedType.isdecimal", isdecimal_impl)
create_masked_unary_identifier_func("MaskedType.istitle", istitle_impl)
create_masked_upper_or_lower("MaskedType.upper", upper_impl)
create_masked_upper_or_lower("MaskedType.lower", lower_impl)
