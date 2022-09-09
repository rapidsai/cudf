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


# String functions
@cuda_decl_registry.register_global(len)
class MaskedStringViewLength(AbstractTemplate):
    """
    provide the length of a cudf::string_view like struct
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(
            args[0].value_type, StringView
        ):
            return nb_signature(MaskedType(size_type), args[0])


@cuda_decl_registry.register_global(operator.contains)
class MaskedStringViewContains(AbstractTemplate):
    """
    return a boolean indicating if a substring is found in a string
    """

    def generic(self, args, kws):
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


@cuda_decl_registry.register_global(len)
class StringLiteralLength(AbstractTemplate):
    """
    provide the length of a python string literal by first
    converting to a cudf::string_view first
    """

    def generic(self, args, kws):
        if isinstance(args[0], types.StringLiteral) and len(args) == 1:
            return nb_signature(size_type, args[0])


def create_masked_binary_attr(attrname, retty):
    class MaskedStringViewBinaryAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(
                MaskedType(retty), MaskedType(string_view), recvr=self.this
            )

    return MaskedStringViewBinaryAttr


def create_masked_identifier_attr(attrname):
    class MaskedStringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(MaskedType(types.boolean), recvr=self.this)

    return MaskedStringViewIdentifierAttr


class MaskedStringViewCount(AbstractTemplate):
    key = "MaskedType.count"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(size_type), MaskedType(string_view), recvr=self.this
        )


@cuda_decl_registry.register_attr
class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_startswith(self, mod):
        return types.BoundFunction(
            create_masked_binary_attr("MaskedType.startswith", types.boolean),
            MaskedType(string_view),
        )

    def resolve_endswith(self, mod):
        return types.BoundFunction(
            create_masked_binary_attr("MaskedType.endswith", types.boolean),
            MaskedType(string_view),
        )

    def resolve_find(self, mod):
        return types.BoundFunction(
            create_masked_binary_attr("MaskedType.find", size_type),
            MaskedType(string_view),
        )

    def resolve_rfind(self, mod):
        return types.BoundFunction(
            create_masked_binary_attr("MaskedType.rfind", size_type),
            MaskedType(string_view),
        )

    def resolve_count(self, mod):
        return types.BoundFunction(
            MaskedStringViewCount, MaskedType(string_view)
        )

    def resolve_isalnum(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isalnum"),
            MaskedType(string_view),
        )

    def resolve_isalpha(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isalpha"),
            MaskedType(string_view),
        )

    def resolve_isdecimal(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isdecimal"),
            MaskedType(string_view),
        )

    def resolve_isdigit(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isdigit"),
            MaskedType(string_view),
        )

    def resolve_islower(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.islower"),
            MaskedType(string_view),
        )

    def resolve_isupper(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isupper"),
            MaskedType(string_view),
        )

    def resolve_isspace(self, mod):
        return types.BoundFunction(
            create_masked_identifier_attr("MaskedType.isspace"),
            MaskedType(string_view),
        )

    def resolve_value(self, mod):
        return string_view

    def resolve_valid(self, mod):
        return types.boolean
