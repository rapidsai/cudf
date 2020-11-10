from typing import List, Any

from cudf.core.column_accessor import ColumnAccessor

class Table(object):

    _data: ColumnAccessor
    _index: "Table"
    
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
