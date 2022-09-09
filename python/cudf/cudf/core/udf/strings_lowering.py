# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import types
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import lower as cuda_lower

from strings_udf._typing import size_type, string_view
from strings_udf.lowering import (
    string_view_contains_impl,
    string_view_count_impl,
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
        context, builder, size_type(string_view), (masked_sv.value,)
    )
    ret.value = result
    ret.valid = masked_sv.valid

    return ret._getvalue()


def create_binary_string_func(op, cuda_func, signature):
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
            nb_signature(*signature),
            (lhs_masked.value, rhs_masked.value),
        )

        ret.value = result
        ret.valid = builder.and_(lhs_masked.valid, rhs_masked.valid)

        return ret._getvalue()

    cuda_lower(op, MaskedType(string_view), MaskedType(string_view))(
        masked_binary_func_impl
    )


create_binary_string_func(
    "MaskedType.startswith",
    string_view_startswith_impl,
    (types.boolean, string_view, string_view),
)

create_binary_string_func(
    "MaskedType.endswith",
    string_view_endswith_impl,
    (types.boolean, string_view, string_view),
)

create_binary_string_func(
    "MaskedType.find",
    string_view_find_impl,
    (size_type, string_view, string_view),
)

create_binary_string_func(
    "MaskedType.rfind",
    string_view_rfind_impl,
    (size_type, string_view, string_view),
)

create_binary_string_func(
    "MaskedType.count",
    string_view_count_impl,
    (size_type, string_view, string_view),
)

create_binary_string_func(
    operator.contains,
    string_view_contains_impl,
    (types.boolean, string_view, string_view),
)


def create_masked_unary_identifier_func(op, cuda_func):
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


create_masked_unary_identifier_func(
    "MaskedType.isalnum", string_view_isalnum_impl
)
create_masked_unary_identifier_func(
    "MaskedType.isalpha", string_view_isalpha_impl
)
create_masked_unary_identifier_func(
    "MaskedType.isdigit", string_view_isdigit_impl
)
create_masked_unary_identifier_func(
    "MaskedType.isupper", string_view_isupper_impl
)
create_masked_unary_identifier_func(
    "MaskedType.islower", string_view_islower_impl
)
create_masked_unary_identifier_func(
    "MaskedType.isspace", string_view_isspace_impl
)
create_masked_unary_identifier_func(
    "MaskedType.isdecimal", string_view_isdecimal_impl
)
