# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.libcudf.replace cimport replace_policy

from .column cimport Column
from .scalar cimport Scalar

ctypedef fused ReplacementType:
    Column
    Scalar
    replace_policy
    # Allowing object is a workaround for
    # https://github.com/cython/cython/issues/5984. See the implementation of
    # replace_nulls for details.
    object


cpdef Column replace_nulls(Column source_column, ReplacementType replacement)

cpdef Column find_and_replace_all(
    Column source_column,
    Column values_to_replace,
    Column replacement_values,
)

cpdef Column clamp(
    Column source_column,
    Scalar lo,
    Scalar hi,
    Scalar lo_replace=*,
    Scalar hi_replace=*,
)

cpdef Column normalize_nans_and_zeros(Column source_column, bool inplace=*)
