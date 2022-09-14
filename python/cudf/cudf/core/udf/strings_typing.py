# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import types
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry

from strings_udf._typing import StringView, size_type, string_view

from cudf.core.udf import masked_typing
from cudf.core.udf._ops import comparison_ops
from cudf.core.udf.masked_typing import MaskedType

masked_typing.MASKED_INIT_MAP[types.pyobject] = string_view
masked_typing.MASKED_INIT_MAP[string_view] = string_view


def _is_valid_string_arg(ty):
    return (
        isinstance(ty, MaskedType) and isinstance(ty.value_type, StringView)
    ) or isinstance(ty, types.StringLiteral)


def register_string_function(func):
    def deco(generic):
        class MaskedStringFunction(AbstractTemplate):
            pass

        MaskedStringFunction.generic = generic
        cuda_decl_registry.register_global(func)(MaskedStringFunction)

    return deco


@register_string_function(len)
def len_typing(self, args, kws):
    if isinstance(args[0], MaskedType) and isinstance(
        args[0].value_type, StringView
    ):
        return nb_signature(MaskedType(size_type), args[0])
    elif isinstance(args[0], types.StringLiteral) and len(args) == 1:
        # literal is always valid
        return nb_signature(size_type, args[0])


def create_masked_binary_attr(attrname, retty):
    class MaskedStringViewBinaryAttr(AbstractTemplate):
        key = f"MaskedType.{attrname}"

        def generic(self, args, kws):
            return nb_signature(
                MaskedType(retty), MaskedType(string_view), recvr=self.this
            )

    return types.BoundFunction(
        MaskedStringViewBinaryAttr,
        MaskedType(string_view),
    )


def create_masked_identifier_attr(attrname):
    class MaskedStringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(MaskedType(types.boolean), recvr=self.this)

    return MaskedStringViewIdentifierAttr


@register_string_function(operator.contains)
def contains_typing(self, args, kws):
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(
            MaskedType(types.boolean),
            MaskedType(string_view),
            MaskedType(string_view),
        )


class MaskedStringViewCmpOp(AbstractTemplate):
    """
    return the boolean result of `cmpop` between to strings
    since the typing is the same for every comparison operator,
    we can reuse this class for all of them.
    """

    def generic(self, args, kws):
        if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
            return nb_signature(
                MaskedType(types.boolean),
                MaskedType(string_view),
                MaskedType(string_view),
            )


class MaskedStringViewCount(AbstractTemplate):
    key = "MaskedType.count"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(size_type), MaskedType(string_view), recvr=self.this
        )


class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_count(self, mod):
        return types.BoundFunction(
            MaskedStringViewCount, MaskedType(string_view)
        )

    def resolve_value(self, mod):
        return string_view

    def resolve_valid(self, mod):
        return types.boolean


identifier_functions = [
    "isupper",
    "islower",
    "isdecimal",
    "isdigit",
    "isspace",
    "isalnum",
    "isalpha",
    "isalnum",
]

binary_bool_funcs = ["startswith", "endswith"]
integer_bool_funcs = ["find", "rfind"]

for func in identifier_functions:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        types.BoundFunction(
            create_masked_identifier_attr(func),
            MaskedType(string_view),
        ),
    )

for func in binary_bool_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(func, types.boolean),
    )

for func in integer_bool_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(func, size_type),
    )

for op in comparison_ops:
    cuda_decl_registry.register_global(op)(MaskedStringViewCmpOp)


cuda_decl_registry.register_attr(MaskedStringViewAttrs)
