# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import cudf
from cudf.core.mixins import NotIterable

if TYPE_CHECKING:
    from cudf.core.index import Index
    from cudf.core.series import Series


class BaseAccessor(NotIterable):
    _parent: Series | Index

    def __init__(self, parent: Series | Index):
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
    ) -> Series | Index: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        expand: bool = False,
        retain_index: bool = True,
    ) -> Series | Index: ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
    ) -> Series | Index | None: ...

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

                if isinstance(self._parent, cudf.Index):
                    idx = self._parent._constructor_expanddim._from_data(table)
                    idx.names = None
                    return idx
                else:
                    df = self._parent._constructor_expanddim._from_data(
                        data=table,
                        index=self._parent.index,
                        attrs=self._parent.attrs,
                    )
                    if len(table) == 0:
                        df._data.rangeindex = True
                    return df
            elif isinstance(self._parent, cudf.Series):
                return cudf.Series._from_column(
                    new_col,
                    name=self._parent.name,
                    index=self._parent.index if retain_index else None,
                    attrs=self._parent.attrs,
                )
            elif isinstance(self._parent, cudf.Index):
                return cudf.Index._from_column(new_col, name=self._parent.name)
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)

    def __setattr__(self, key, value):
        if key in {"_parent", "_column"}:
            super().__setattr__(key, value)
        else:
            raise AttributeError(f"You cannot add any new attribute '{key}'")
