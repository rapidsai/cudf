# Copyright (c) 2020, NVIDIA CORPORATION.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, overload

from typing_extensions import Literal

import cudf

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase

ParentType = Union["cudf.Series", "cudf.BaseIndex"]


class ColumnMethodsMixin:
    _column: ColumnBase
    _parent: Optional[Union["cudf.Series", "cudf.BaseIndex"]]

    def __init__(
        self,
        column: ColumnBase,
        parent: Union["cudf.Series", "cudf.BaseIndex"] = None,
    ):
        self._column = column
        self._parent = parent

    @overload
    def _return_or_inplace(
        self, new_col, inplace: Literal[False], expand=False, retain_index=True
    ) -> Union["cudf.Series", "cudf.BaseIndex"]:
        ...

    @overload
    def _return_or_inplace(
        self, new_col, expand: bool = False, retain_index: bool = True
    ) -> Union["cudf.Series", "cudf.BaseIndex"]:
        ...

    @overload
    def _return_or_inplace(
        self, new_col, inplace: Literal[True], expand=False, retain_index=True
    ) -> None:
        ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
    ) -> Optional[Union["cudf.Series", "cudf.BaseIndex"]]:
        ...

    def _return_or_inplace(
        self, new_col, inplace=False, expand=False, retain_index=True
    ):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        if inplace:
            if self._parent is not None:
                self._parent._mimic_inplace(
                    self._parent.__class__._from_table(
                        cudf._lib.table.Table({self._parent.name: new_col})
                    ),
                    inplace=True,
                )
                return None
            else:
                self._column._mimic_inplace(new_col, inplace=True)
                return None
        else:
            if self._parent is None:
                return new_col
            if expand or isinstance(
                self._parent, (cudf.DataFrame, cudf.MultiIndex)
            ):
                # This branch indicates the passed as new_col
                # is a Table
                table = new_col

                if isinstance(self._parent, cudf.BaseIndex):
                    idx = self._parent._constructor_expanddim._from_table(
                        table=table
                    )
                    idx.names = None
                    return idx
                else:
                    return self._parent._constructor_expanddim(
                        data=table._data, index=self._parent.index
                    )
            elif isinstance(self._parent, cudf.Series):
                if retain_index:
                    return cudf.Series(
                        new_col,
                        name=self._parent.name,
                        index=self._parent.index,
                    )
                else:
                    return cudf.Series(new_col, name=self._parent.name)
            elif isinstance(self._parent, cudf.BaseIndex):
                return cudf.core.index.as_index(
                    new_col, name=self._parent.name
                )
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)
