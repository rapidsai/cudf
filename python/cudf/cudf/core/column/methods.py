# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Literal, Union, overload

import cudf
import cudf.core.column
import cudf.core.column_accessor
from cudf.utils.utils import NotIterable

ParentType = Union["cudf.Series", "cudf.core.index.Index"]


class ColumnMethods(NotIterable):
    _parent: ParentType

    def __init__(self, parent: ParentType):
        self._parent = parent
        self._column = self._parent._column

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[True],
        expand: bool = False,
        retain_index: bool = True,
    ) -> None: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[False],
        expand: bool = False,
        retain_index: bool = True,
    ) -> ParentType: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        expand: bool = False,
        retain_index: bool = True,
    ) -> ParentType: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
    ) -> ParentType | None: ...

    def _return_or_inplace(
        self, new_col, inplace=False, expand=False, retain_index=True
    ):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        if inplace:
            self._parent._mimic_inplace(
                type(self._parent)._from_column(
                    new_col, name=self._parent.name
                ),
                inplace=True,
            )
            return None
        else:
            if expand:
                # This branch indicates the passed as new_col
                # is a Table
                table = new_col

                if isinstance(self._parent, cudf.BaseIndex):
                    idx = self._parent._constructor_expanddim._from_data(table)
                    idx.names = None
                    return idx
                else:
                    return self._parent._constructor_expanddim._from_data(
                        data=table, index=self._parent.index
                    )
            elif isinstance(self._parent, cudf.Series):
                return cudf.Series._from_column(
                    new_col,
                    name=self._parent.name,
                    index=self._parent.index if retain_index else None,
                )
            elif isinstance(self._parent, cudf.BaseIndex):
                return cudf.Index._from_column(new_col, name=self._parent.name)
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)
