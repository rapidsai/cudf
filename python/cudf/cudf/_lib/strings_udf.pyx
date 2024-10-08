# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.strings_udf cimport (
    get_cuda_build_version as cpp_get_cuda_build_version,
)

from cudf._lib.column import f


def get_cuda_build_version():
    return cpp_get_cuda_build_version()
