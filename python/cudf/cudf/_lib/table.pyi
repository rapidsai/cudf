# Copyright (c) 2021, NVIDIA CORPORATION.

from typing import List, Any, Optional, TYPE_CHECKING

import cudf

class Table(object):
    _data: cudf.core.column_accessor.ColumnAccessor
    _index: Optional[cudf.core.index.Index]

    def __init__(self, data: object = None, index: object = None) -> None: ...

    @property
    def _num_columns(self) -> int: ...

    @property
    def _num_indices(self) -> int: ...

    @property
    def _num_rows(self) -> int: ...

    @property
    def _column_names(self) -> List[Any]: ...

    @property
    def _index_names(self) -> List[Any]: ...

    @property
    def _columns(self) -> List[Any]: ... # TODO: actually, a list of columns
