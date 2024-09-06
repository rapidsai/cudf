# Copyright (c) 2023-2024, NVIDIA CORPORATION.

# If libcudf was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcudf
except ModuleNotFoundError:
    pass
else:
    libcudf.load_library()
    del libcudf

from . import (
    aggregation,
    binaryop,
    column_factories,
    concatenate,
    copying,
    datetime,
    experimental,
    expressions,
    filling,
    groupby,
    interop,
    io,
    join,
    lists,
    merge,
    null_mask,
    quantiles,
    reduce,
    replace,
    reshape,
    rolling,
    round,
    search,
    sorting,
    stream_compaction,
    strings,
    traits,
    transform,
    types,
    unary,
)
from .column import Column
from .gpumemoryview import gpumemoryview
from .scalar import Scalar
from .table import Table
from .types import DataType, MaskState, TypeId

__all__ = [
    "Column",
    "DataType",
    "MaskState",
    "Scalar",
    "Table",
    "TypeId",
    "aggregation",
    "binaryop",
    "column_factories",
    "concatenate",
    "copying",
    "datetime",
    "experimental",
    "expressions",
    "filling",
    "gpumemoryview",
    "groupby",
    "interop",
    "join",
    "lists",
    "merge",
    "null_mask",
    "quantiles",
    "reduce",
    "replace",
    "reshape",
    "rolling",
    "round",
    "search",
    "stream_compaction",
    "strings",
    "sorting",
    "traits",
    "transform",
    "types",
    "unary",
]
