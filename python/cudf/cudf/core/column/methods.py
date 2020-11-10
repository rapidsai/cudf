# Copyright (c) 2020, NVIDIA CORPORATION.

from typing import Optional, Union

import cudf


class ColumnMethodsMixin:
    def __init__(
        self, column, parent: Union["cudf.Series", "cudf.Index"] = None
    ):
        self._column = column
        self._parent = parent

    def _return_or_inplace(
        self, new_col, **kwargs
    ) -> Optional[Union["cudf.Series", "cudf.Index"]]:
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        inplace = kwargs.get("inplace", False)

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
            expand = kwargs.get("expand", False) or isinstance(
                self._parent, (cudf.DataFrame, cudf.MultiIndex)
            )
            if expand:
                # This branch indicates the passed as new_col
                # is a Table
                table = new_col

                if isinstance(self._parent, cudf.Index):
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
                retain_index = kwargs.get("retain_index", True)
                if retain_index:
                    return cudf.Series(
                        new_col,
                        name=self._parent.name,
                        index=self._parent.index,
                    )
                else:
                    return cudf.Series(new_col, name=self._parent.name)
            elif isinstance(self._parent, cudf.Index):
                return cudf.core.index.as_index(
                    new_col, name=self._parent.name
                )
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)
