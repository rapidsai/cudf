# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import types
from numba.core import cgutils
from numba.cuda.cudaimpl import lower as cuda_lower

from strings_udf._typing import string_view
from strings_udf.lowering import (
    string_view_contains_impl,
    string_view_endswith_impl,
    string_view_find_impl,
    string_view_isalnum_impl,
    string_view_isalpha_impl,
    string_view_isdecimal_impl,
    string_view_isdigit_impl,
    string_view_islower_impl,
    string_view_isspace_impl,
    string_view_isupper_impl,
    string_view_len_impl,
    string_view_rfind_impl,
    string_view_startswith_impl,
)

from cudf.core.udf.masked_typing import MaskedType


@cuda_lower(len, MaskedType(string_view))
def masked_string_view_len_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    result = string_view_len_impl(
        context, builder, types.int32(string_view), (masked_sv.value,)
    )
    ret.value = result
    ret.valid = masked_sv.valid

    return ret._getvalue()


@cuda_lower(
    "MaskedType.startswith", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_startswith_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_startswith_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value, masked_sv_substr.value),
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower(
    "MaskedType.endswith", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_endswith_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_endswith_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value, masked_sv_substr.value),
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower(
    "MaskedType.find", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_find_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_find_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value, masked_sv_substr.value),
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower(
    "MaskedType.rfind", MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_rfind_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_rfind_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value, masked_sv_substr.value),
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower(
    operator.contains, MaskedType(string_view), MaskedType(string_view)
)
def masked_string_view_contains_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )
    masked_sv_substr = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[1]
    )
    result = string_view_contains_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value, masked_sv_substr.value),
    )

    ret.value = result
    ret.valid = builder.and_(masked_sv_str.valid, masked_sv_substr.valid)
    return ret._getvalue()


@cuda_lower("MaskedType.isalnum", MaskedType(string_view))
def masked_string_view_isalnum_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isalnum_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.isalpha", MaskedType(string_view))
def masked_string_view_isalpha_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isalpha_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.isdigit", MaskedType(string_view))
def masked_string_view_isdigit_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isdigit_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.isdecimal", MaskedType(string_view))
def masked_string_view_isdecimal_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isdecimal_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.isupper", MaskedType(string_view))
def masked_string_view_isupper_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isupper_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.islower", MaskedType(string_view))
def masked_string_view_islower_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_islower_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()


@cuda_lower("MaskedType.isspace", MaskedType(string_view))
def masked_string_view_isspace_impl(context, builder, sig, args):
    ret = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    masked_sv_ty = sig.args[0]
    masked_sv_str = cgutils.create_struct_proxy(masked_sv_ty)(
        context, builder, value=args[0]
    )

    result = string_view_isspace_impl(
        context,
        builder,
        types.boolean(string_view, string_view),
        (masked_sv_str.value,),
    )

    ret.value = result
    ret.valid = masked_sv_str.valid
    return ret._getvalue()
