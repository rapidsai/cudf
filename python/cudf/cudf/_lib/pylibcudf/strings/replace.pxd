# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.scalar cimport Scalar
from cudf._lib.pylibcudf.strings.types cimport ColumnOrScalar


cpdef Column replace(
    Column input,
    ColumnOrScalar target,
    ColumnOrScalar repl,
    size_type maxrepl = *
)
cpdef Column replace_slice(
    Column input,
    Scalar repl = *,
    size_type start = *,
    size_type stop = *
)
