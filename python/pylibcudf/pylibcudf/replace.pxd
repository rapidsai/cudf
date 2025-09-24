# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.replace cimport replace_policy
from rmm.pylibrmm.stream cimport Stream

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


cpdef Column replace_nulls(
    Column source_column,
    ReplacementType replacement,
    Stream stream = *,
)

cpdef Column find_and_replace_all(
    Column source_column,
    Column values_to_replace,
    Column replacement_values,
    Stream stream = *
)

cpdef Column clamp(
    Column source_column,
    Scalar lo,
    Scalar hi,
    Scalar lo_replace=*,
    Scalar hi_replace=*,
    Stream stream = *
)

cpdef Column normalize_nans_and_zeros(
    Column source_column,
    bool inplace=*,
    Stream stream = *,
)
