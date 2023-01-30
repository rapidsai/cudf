# Copyright (c) 2023, NVIDIA CORPORATION.
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudaimpl import lower as cuda_lower

from strings_udf._typing import StringView, string_view, udf_string
from strings_udf.lowering import cast_string_view_to_udf_string


def sv_to_udf_str(sv):
    pass


@cuda_decl_registry.register_global(sv_to_udf_str)
class StringViewToUDFStringDecl(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], StringView) and len(args) == 1:
            return nb_signature(udf_string, string_view)


@cuda_lower(sv_to_udf_str, string_view)
def sv_to_udf_str_testing_lowering(context, builder, sig, args):
    return cast_string_view_to_udf_string(
        context, builder, sig.args[0], sig.return_type, args[0]
    )
