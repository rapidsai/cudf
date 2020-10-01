# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf


class ColumnMethodsMixin:
    def _return_or_inplace(self, new_col, **kwargs):
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
            else:
                self._column._mimic_inplace(new_col, inplace=True)
        else:
            expand = kwargs.get("expand", False)
            if expand or isinstance(
                self._parent, (cudf.DataFrame, cudf.MultiIndex)
            ):
                # This branch indicates the passed as new_col
                # is actually a table-like data
                table = new_col

                if isinstance(table, cudf._lib.table.Table):
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
                else:
                    return self._parent._constructor_expanddim(
                        {index: value for index, value in enumerate(table)},
                        index=self._parent.index,
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
                if self._parent is None:
                    return new_col
                else:
                    return self._parent._mimic_inplace(new_col, inplace=False)
