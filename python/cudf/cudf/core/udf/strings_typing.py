# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

from numba import types
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry

from strings_udf._typing import (
    StringView,
    bool_binary_funcs,
    id_unary_funcs,
    int_binary_funcs,
    size_type,
    string_return_attrs,
    string_unary_funcs,
    string_view,
    udf_string,
)

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
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to associate a signature with a function or
    operator to be overloaded.
    """

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
        return nb_signature(size_type, args[0])


@register_string_function(operator.add)
def concat_typing(self, args, kws):
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(
            MaskedType(udf_string),
            MaskedType(string_view),
            MaskedType(string_view),
        )


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


for op in comparison_ops:
    cuda_decl_registry.register_global(op)(MaskedStringViewCmpOp)


def create_masked_binary_attr(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a binary function of two masked
    string objects as an attribute of one, e.g. `string.func(other)`.
    """

    class MaskedStringViewBinaryAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(
                MaskedType(retty), MaskedType(string_view), recvr=self.this
            )

    def attr(self, mod):
        return types.BoundFunction(
            MaskedStringViewBinaryAttr,
            MaskedType(string_view),
        )

    return attr


def create_masked_unary_attr(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a unary function of a masked
    string object as an attribute, e.g. `string.func()`.
    """

    class MaskedStringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(MaskedType(retty), recvr=self.this)

    def attr(self, mod):
        return types.BoundFunction(
            MaskedStringViewIdentifierAttr,
            MaskedType(string_view),
        )

    return attr


class MaskedStringViewCount(AbstractTemplate):
    key = "MaskedType.count"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(size_type), MaskedType(string_view), recvr=self.this
        )


class MaskedStringViewReplace(AbstractTemplate):
    key = "MaskedType.replace"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(udf_string),
            MaskedType(string_view),
            MaskedType(string_view),
            recvr=self.this,
        )


class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_replace(self, mod):
        return types.BoundFunction(
            MaskedStringViewReplace, MaskedType(string_view)
        )

    def resolve_count(self, mod):
        return types.BoundFunction(
            MaskedStringViewCount, MaskedType(string_view)
        )

    def resolve_value(self, mod):
        return string_view

    def resolve_valid(self, mod):
        return types.boolean


# Build attributes for `MaskedType(string_view)`
for func in bool_binary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", types.boolean),
    )

for func in int_binary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", size_type),
    )

for func in string_return_attrs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", udf_string),
    )

for func in id_unary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_unary_attr(f"MaskedType.{func}", types.boolean),
    )

for func in string_unary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_unary_attr(f"MaskedType.{func}", udf_string),
    )

cuda_decl_registry.register_attr(MaskedStringViewAttrs)
