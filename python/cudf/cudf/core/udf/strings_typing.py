# Copyright (c) 2022, NVIDIA CORPORATION.


#make_attribute_wrapper(StringView, "data", "data")
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)

import operator

from cudf.core.udf.masked_typing import MaskedType

from strings_udf._typing import StringView, string_view
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba import types
from numba.core.typing import signature as nb_signature

from cudf.core.udf._ops import comparison_ops

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
            return nb_signature(MaskedType(types.int32), args[0])

@cuda_decl_registry.register_global(operator.contains)
class MaskedStringViewContains(AbstractTemplate):
    """
    return a boolean indicating if a substring is found in a string
    """

    def generic(self, args, kws):
        if (isinstance(args[0], MaskedType) and isinstance(
            args[0].value_type, StringView
        ) or isinstance(args[0], types.StringLiteral)) and (isinstance(args[1], MaskedType) and isinstance(
            args[1].value_type, StringView
        ) or isinstance(args[1], types.StringLiteral)):
            return nb_signature(MaskedType(types.boolean), MaskedType(string_view), MaskedType(string_view))

class MaskedStringViewCmpOp(AbstractTemplate):
    """
    return the boolean result of `cmpop` between to strings
    since the typing is the same for every comparison operator,
    we can reuse this class for all of them.
    """

    def generic(self, args, kws):
        if (isinstance(args[0], MaskedType) and isinstance(
            args[0].value_type, StringView
        ) or isinstance(args[0], types.StringLiteral)) and (isinstance(args[1], MaskedType) and isinstance(
            args[1].value_type, StringView
        ) or isinstance(args[1], types.StringLiteral)):
            return nb_signature(MaskedType(types.boolean), MaskedType(string_view), MaskedType(string_view))

@cuda_decl_registry.register_global(len)
class StringLiteralLength(AbstractTemplate):
    """
    provide the length of a python string literal by first
    converting to a cudf::string_view first
    """

    def generic(self, args, kws):
        if isinstance(args[0], types.StringLiteral) and len(args) == 1:
            return nb_signature(types.int32, args[0])

class MaskedStringViewStartsWith(AbstractTemplate):
    key = "MaskedType.startswith"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(types.boolean), MaskedType(string_view), recvr=self.this
        )

class MaskedStringViewEndsWith(AbstractTemplate):
    key = "MaskedType.endswith"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(types.boolean), MaskedType(string_view), recvr=self.this
        )

class MaskedStringViewFind(AbstractTemplate):
    key = "MaskedType.find"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(types.int32), MaskedType(string_view), recvr=self.this
        )

class MaskedStringViewRFind(AbstractTemplate):
    key = "MaskedType.rfind"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(types.int32), MaskedType(string_view), recvr=self.this
        )

@cuda_decl_registry.register_attr
class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_startswith(self, mod):
        return types.BoundFunction(
            MaskedStringViewStartsWith, MaskedType(string_view)
        )

    def resolve_endswith(self, mod):
        return types.BoundFunction(
            MaskedStringViewEndsWith, MaskedType(string_view)
        )

    def resolve_find(self, mod):
        return types.BoundFunction(
            MaskedStringViewFind, MaskedType(string_view)
        )

    def resolve_rfind(self, mod):
        return types.BoundFunction(
            MaskedStringViewRFind, MaskedType(string_view)
        )

    def resolve_value(self, mod):
        return string_view

    def resolve_valid(self, mod):
        return types.boolean

for op in comparison_ops:
    cuda_decl_registry.register_global(op)(MaskedStringViewCmpOp)
