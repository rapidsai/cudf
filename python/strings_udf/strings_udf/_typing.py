# Copyright (c) 2022, NVIDIA CORPORATION.

from numba import cuda, types
from numba.core.extending import models, register_model
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry

import operator

import llvmlite.binding as ll
from numba.cuda.cudadrv import nvvm
from numba.core.datamodel import default_manager

import operator

data_layout = nvvm.data_layout

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
        #  size_type _bytes{};
        #  Number of bytes in _data for this string
        ("bytes", types.int32),
        #  mutable size_type _length{};
        #  Number of characters in this string (computed)
        ("length", types.int32),
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
        ("m_bytes", types.int32),
        ("m_size", types.int32),
    )

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, self._members)


any_string_ty = (StringView, DString, types.StringLiteral)
string_view = StringView()


class StrViewArgHandler:
    """
    As part of Numbas preprocessing step incoming function arguments are
    modified based on the associated type for that argument that was used
    to JIT the kernel. However it only knows how to handle built in array
    types natively. With string UDFs, the jitted type is string_view*,
    which numba does not know how to handle.

    This small piece of code implements the necessary handling. Really all
    it does is funnel the handling of string_view* to the handling
    of raw pointer arguments, which numba knows how to use.

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
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and len(args) == 1
        ):
            # length:
            # string_view -> int32
            # dstring -> int32
            # literal -> int32
            return nb_signature(types.int32, args[0])


_string_view_len = cuda.declare_device("len", types.int32(types.CPointer(string_view)))

@cuda_decl_registry.register_global(operator.contains)
class StringViewContains(AbstractTemplate):
    """
    Return True if a string view contains a substring view
    """

    def generic(self, args, kws):
        if isinstance(
            args[0], (StringView, DString, types.StringLiteral)
        ) and isinstance(args[1], (StringView, DString, types.StringLiteral)):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_contains = cuda.declare_device(
    "contains", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)

@cuda_decl_registry.register_global(operator.eq)
class StringViewEq(AbstractTemplate):
    """
    Compare two cudf::string_view with ==
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_eq = cuda.declare_device(
    "eq", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


@cuda_decl_registry.register_global(operator.ne)
class StringViewNe(AbstractTemplate):
    """
    Compare two cudf::string_view with !=
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_ne = cuda.declare_device(
    "ne", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


@cuda_decl_registry.register_global(operator.ge)
class StringViewGe(AbstractTemplate):
    """
    Compare two cudf::string_view with >=
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_ge = cuda.declare_device(
    "ge", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


@cuda_decl_registry.register_global(operator.le)
class StringViewLe(AbstractTemplate):
    """
    Compare two cudf::string_view with <=
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_le = cuda.declare_device(
    "le", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


@cuda_decl_registry.register_global(operator.gt)
class StringViewGt(AbstractTemplate):
    """
    Compare two cudf::string_view with >
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_gt = cuda.declare_device(
    "gt", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


@cuda_decl_registry.register_global(operator.lt)
class StringViewLt(AbstractTemplate):
    """
    Compare two cudf::string_view with <
    """

    def generic(self, args, kws):
        if (
            isinstance(args[0], (StringView, DString, types.StringLiteral))
            and isinstance(args[1], (StringView, DString, types.StringLiteral))
            and len(args) == 2
        ):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_lt = cuda.declare_device(
    "lt", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)


def starts_with(st, substr):
    return st.startswith(substr)


@cuda_decl_registry.register_global(starts_with)
class StringViewStartsWith(AbstractTemplate):
    """
    return True if a stringview starts with a substring
    """

    def generic(self, args, kws):
        if isinstance(args[0], (any_string_ty)) and isinstance(args[1], any_string_ty):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_startswith = cuda.declare_device(
    "startswith",
    types.boolean(types.CPointer(string_view), types.CPointer(string_view)),
)


def ends_with(st, substr):
    return st.endswith(substr)


@cuda_decl_registry.register_global(ends_with)
class StringViewEndsWith(AbstractTemplate):
    """
    return True if a stringview ends with a substring
    """

    def generic(self, args, kws):
        if isinstance(args[0], (any_string_ty)) and isinstance(args[1], any_string_ty):
            return nb_signature(types.boolean, string_view, string_view)


_string_view_endswith = cuda.declare_device(
    "endswith", types.boolean(types.CPointer(string_view), types.CPointer(string_view))
)

def find(st, substr):
    return st.find(substr)

@cuda_decl_registry.register_global(find)
class StringViewFind(AbstractTemplate):
    """
    Return the index of a substring within a stringview
    """

    def generic(self, args, kws):
        if isinstance(args[0], (any_string_ty)) and isinstance(args[1], any_string_ty):
            return nb_signature(types.int32, string_view, string_view)

_string_view_find = cuda.declare_device(
    "find",
    types.int32(types.CPointer(string_view), types.CPointer(string_view))
)

def rfind(st, substr):
    return st.rfind(substr)

@cuda_decl_registry.register_global(rfind)
class StringViewFind(AbstractTemplate):
    """
    Return the index of a substring within a stringview
    """

    def generic(self, args, kws):
        if isinstance(args[0], (any_string_ty)) and isinstance(args[1], any_string_ty):
            return nb_signature(types.int32, string_view, string_view)

_string_view_rfind = cuda.declare_device(
    "rfind",
    types.int32(types.CPointer(string_view), types.CPointer(string_view))
)

def isdigit(st):
    return st.isdigit()

@cuda_decl_registry.register_global(isdigit)
class StringViewIsdigit(AbstractTemplate):
    """
    Return True if the string is all numeric characters else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isdigit = cuda.declare_device(
    "pyisdigit",
    types.boolean(types.CPointer(string_view), types.int64)
)

def isalnum(st):
    return st.isalnum()

@cuda_decl_registry.register_global(isalnum)
class StringViewIsalnum(AbstractTemplate):
    """
    Return True if the string is all alphanumeric characters else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isalnum = cuda.declare_device(
    "pyisalnum",
    types.boolean(types.CPointer(string_view), types.int64)
)

def isdecimal(st):
    return st.isdecimal()

@cuda_decl_registry.register_global(isdecimal)
class StringViewIsdecimal(AbstractTemplate):
    """
    Return True if the string is all decimal characters else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isdecimal = cuda.declare_device(
    "pyisdecimal",
    types.boolean(types.CPointer(string_view), types.int64)
)

def isnumeric(st):
    return st.isnumeric()

@cuda_decl_registry.register_global(isnumeric)
class StringViewIsnumeric(AbstractTemplate):
    """
    Return True if the string represents a valid number else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isnumeric = cuda.declare_device(
    "pyisnumeric",
    types.boolean(types.CPointer(string_view), types.int64)
)

def isspace(st):
    return st.isspace()

@cuda_decl_registry.register_global(isspace)
class StringViewIsspace(AbstractTemplate):
    """
    Return True if the string is all white space else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isspace = cuda.declare_device(
    "pyisspace",
    types.boolean(types.CPointer(string_view), types.int64)
)

def isupper(st):
    return st.isupper()

@cuda_decl_registry.register_global(isupper)
class StringViewIsupper(AbstractTemplate):
    """
    Return True if the string's alphabetic characters are all uppercase else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_isupper = cuda.declare_device(
    "pyisupper",
    types.boolean(types.CPointer(string_view), types.int64)
)

def islower(st):
    return st.islower()

@cuda_decl_registry.register_global(islower)
class StringViewIslower(AbstractTemplate):
    """
    Return True if the string's alphabetic characters are all lowercase else false
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            return nb_signature(types.boolean, string_view)

_string_view_islower = cuda.declare_device(
    "pyislower",
    types.boolean(types.CPointer(string_view), types.int64)
)


def count(st, substr):
    return st.count(substr)

@cuda_decl_registry.register_global(count)
class StringViewCount(AbstractTemplate):
    """
    Return the number of non-overlapping occurences of a substring within a string
    """
    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and isinstance(args[1], any_string_ty):
            return nb_signature(types.int32, string_view, string_view)

_string_view_count = cuda.declare_device(
    "pycount",
    types.int32(types.CPointer(string_view), types.CPointer(string_view))
)
