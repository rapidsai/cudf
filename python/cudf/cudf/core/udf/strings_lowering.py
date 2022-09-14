# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import types
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import lower as cuda_lower

from strings_udf._typing import size_type, string_view
from strings_udf.lowering import (
    contains_impl,
    count_impl,
    endswith_impl,
    find_impl,
    isalnum_impl,
    isalpha_impl,
    isdecimal_impl,
    isdigit_impl,
    islower_impl,
    isspace_impl,
    isupper_impl,
    len_impl,
    rfind_impl,
    startswith_impl,
)

from cudf.core.udf.masked_typing import MaskedType


@cuda_lower(len, MaskedType(string_view))
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


def create_binary_string_func(op, cuda_func, retty):
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


create_binary_string_func(
    "MaskedType.startswith",
    startswith_impl,
    types.boolean,
)
create_binary_string_func("MaskedType.endswith", endswith_impl, types.boolean)
create_binary_string_func("MaskedType.find", find_impl, size_type)
create_binary_string_func("MaskedType.rfind", rfind_impl, size_type)
create_binary_string_func("MaskedType.count", count_impl, size_type)
create_binary_string_func(operator.contains, contains_impl, types.boolean)


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


create_masked_unary_identifier_func("MaskedType.isalnum", isalnum_impl)
create_masked_unary_identifier_func("MaskedType.isalpha", isalpha_impl)
create_masked_unary_identifier_func("MaskedType.isdigit", isdigit_impl)
create_masked_unary_identifier_func("MaskedType.isupper", isupper_impl)
create_masked_unary_identifier_func("MaskedType.islower", islower_impl)
create_masked_unary_identifier_func("MaskedType.isspace", isspace_impl)
create_masked_unary_identifier_func("MaskedType.isdecimal", isdecimal_impl)
