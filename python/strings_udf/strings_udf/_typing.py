# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

import llvmlite.binding as ll
from numba import types
from numba.core.datamodel import default_manager
from numba.core.extending import models, register_model
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudadrv import nvvm

data_layout = nvvm.data_layout

# libcudf size_type
size_type = types.int32

# workaround for numba < 0.56
if isinstance(data_layout, dict):
    data_layout = data_layout[64]
target_data = ll.create_target_data(data_layout)


# String object definitions
class DString(types.Type):
    def __init__(self):
        super().__init__(name="dstring")
        llty = default_manager[self].get_value_type()
        self.size_bytes = llty.get_abi_size(target_data)


class StringView(types.Type):
    def __init__(self):
        super().__init__(name="string_view")
        llty = default_manager[self].get_value_type()
        self.size_bytes = llty.get_abi_size(target_data)


@register_model(StringView)
class stringview_model(models.StructModel):
    # from string_view.hpp:
    _members = (
        # const char* _data{}
        # Pointer to device memory contain char array for this string
        ("data", types.CPointer(types.char)),
        # size_type _bytes{};
        # Number of bytes in _data for this string
        ("bytes", size_type),
        # mutable size_type _length{};
        # Number of characters in this string (computed)
        ("length", size_type),
    )

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, self._members)


@register_model(DString)
class dstring_model(models.StructModel):
    # from dstring.hpp:
    # private:
    #   char* m_data{};
    #   cudf::size_type m_bytes{};
    #   cudf::size_type m_size{};

    _members = (
        ("m_data", types.CPointer(types.char)),
        ("m_bytes", size_type),
        ("m_size", size_type),
    )

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, self._members)


any_string_ty = (StringView, DString, types.StringLiteral)
string_view = StringView()


class StrViewArgHandler:
    """
    As part of Numba's preprocessing step, incoming function arguments are
    modified based on the associated type for that argument that was used
    to JIT the kernel. However it only knows how to handle built in array
    types natively. With string UDFs, the jitted type is string_view*,
    which numba does not know how to handle.

    This class converts string_view* to raw pointer arguments, which Numba
    knows how to use.

    See numba.cuda.compiler._prepare_args for details.
    """

    def prepare_args(self, ty, val, **kwargs):
        if isinstance(ty, types.CPointer) and isinstance(ty.dtype, StringView):
            return types.uint64, val.ptr
        else:
            return ty, val


str_view_arg_handler = StrViewArgHandler()


# String functions
@cuda_decl_registry.register_global(len)
class StringLength(AbstractTemplate):
    """
    provide the length of a cudf::string_view like struct
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            # length:
            # string_view -> int32
            # dstring -> int32
            # literal -> int32
            return nb_signature(size_type, args[0])


def register_stringview_binaryop(op, retty):
    class StringViewBinaryOp(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], any_string_ty) and isinstance(
                args[1], any_string_ty
            ):
                return nb_signature(retty, string_view, string_view)

    cuda_decl_registry.register_global(op)(StringViewBinaryOp)


register_stringview_binaryop(operator.eq, types.boolean)
register_stringview_binaryop(operator.ne, types.boolean)
register_stringview_binaryop(operator.lt, types.boolean)
register_stringview_binaryop(operator.gt, types.boolean)
register_stringview_binaryop(operator.le, types.boolean)
register_stringview_binaryop(operator.ge, types.boolean)
register_stringview_binaryop(operator.contains, types.boolean)


def create_binary_attr(attrname, retty):
    class StringViewBinaryAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(retty, string_view, recvr=self.this)

    return StringViewBinaryAttr


def create_identifier_attr(attrname):
    class StringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(types.boolean, recvr=self.this)

    return StringViewIdentifierAttr


class StringViewCount(AbstractTemplate):
    key = "StringView.count"

    def generic(self, args, kws):
        return nb_signature(size_type, string_view, recvr=self.this)


@cuda_decl_registry.register_attr
class StringViewAttrs(AttributeTemplate):
    key = string_view

    def resolve_startswith(self, mod):
        return types.BoundFunction(
            create_binary_attr("StringView.startswith", types.boolean),
            string_view,
        )

    def resolve_endswith(self, mod):
        return types.BoundFunction(
            create_binary_attr("StringView.endswith", types.boolean),
            string_view,
        )

    def resolve_find(self, mod):
        return types.BoundFunction(
            create_binary_attr("StringView.find", size_type), string_view
        )

    def resolve_rfind(self, mod):
        return types.BoundFunction(
            create_binary_attr("StringView.rfind", size_type), string_view
        )

    def resolve_isalnum(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isalnum"), string_view
        )

    def resolve_isalpha(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isalpha"), string_view
        )

    def resolve_isdecimal(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isdecimal"), string_view
        )

    def resolve_isdigit(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isdigit"), string_view
        )

    def resolve_isnumeric(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isnumeric"), string_view
        )

    def resolve_islower(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.islower"), string_view
        )

    def resolve_isupper(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isupper"), string_view
        )

    def resolve_isspace(self, mod):
        return types.BoundFunction(
            create_identifier_attr("StringView.isspace"), string_view
        )

    def resolve_count(self, mod):
        return types.BoundFunction(StringViewCount, string_view)
