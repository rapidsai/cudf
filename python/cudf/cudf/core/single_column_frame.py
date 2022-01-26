# Copyright (c) 2021, NVIDIA CORPORATION.
"""Base class for Frame types that only have a single column."""

from __future__ import annotations

import warnings
from typing import Any, Dict, MutableMapping, Optional, Tuple, TypeVar, Union

import cupy
import numpy as np
import pandas as pd

import cudf
from cudf._typing import Dtype
from cudf.api.types import _is_scalar_or_zero_d_array
from cudf.core.column import ColumnBase, as_column
from cudf.core.frame import Frame

T = TypeVar("T", bound="Frame")


class SingleColumnFrame(Frame):
    """A one-dimensional frame.

    Frames with only a single column share certain logic that is encoded in
    this class.
    """

    _SUPPORT_AXIS_LOOKUP = {
        0: 0,
        None: 0,
        "index": 0,
    }

    def _reduce(
        self, op, axis=None, level=None, numeric_only=None, **kwargs,
    ):
        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )
        return getattr(self._column, op)(**kwargs)

    def _scan(self, op, axis=None, *args, **kwargs):
        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        return super()._scan(op, axis=axis, *args, **kwargs)

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[cudf.core.index.BaseIndex] = None,
        name: Any = None,
    ):

        out = super()._from_data(data, index)
        if name is not None:
            out.name = name
        return out

    @property
    def name(self):
        """Get the name of this object."""
        return next(iter(self._data.names))

    @name.setter
    def name(self, value):
        self._data[value] = self._data.pop(self.name)

    @property
    def ndim(self):
        """Get the dimensionality (always 1 for single-columned frames)."""
        return 1

    @property
    def shape(self):
        """Get a tuple representing the dimensionality of the Index."""
        return (len(self),)

    def __iter__(self):
        # Iterating over a GPU object is not efficient and hence not supported.
        # Consider using ``.to_arrow()``, ``.to_pandas()`` or ``.values_host``
        # if you wish to iterate over the values.
        cudf.utils.utils.raise_iteration_error(obj=self)

    def __bool__(self):
        raise TypeError(
            f"The truth value of a {type(self)} is ambiguous. Use "
            "a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    @property
    def _num_columns(self):
        return 1

    @property
    def _column(self):
        return self._data[self.name]

    @_column.setter
    def _column(self, value):
        self._data[self.name] = value

    @property
    def values(self):  # noqa: D102
        return self._column.values

    @property
    def values_host(self):  # noqa: D102
        return self._column.values_host

    def to_cupy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = True,
        na_value=None,
    ) -> cupy.ndarray:  # noqa: D102
        return super().to_cupy(dtype, copy, na_value).flatten()

    def to_numpy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = True,
        na_value=None,
    ) -> np.ndarray:  # noqa: D102
        return super().to_numpy(dtype, copy, na_value).flatten()

    def tolist(self):  # noqa: D102

        raise TypeError(
            "cuDF does not support conversion to host memory "
            "via the `tolist()` method. Consider using "
            "`.to_arrow().to_pylist()` to construct a Python list."
        )

    to_list = tolist

    # TODO: When this method is removed we can also remove
    # ColumnBase.to_gpu_array.
    def to_gpu_array(self, fillna=None):  # noqa: D102
        warnings.warn(
            "The to_gpu_array method will be removed in a future cuDF "
            "release. Consider using `to_cupy` instead.",
            FutureWarning,
        )
        return self._column.to_gpu_array(fillna=fillna)

    @classmethod
    def from_arrow(cls, array):
        """Create from PyArrow Array/ChunkedArray.

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray
            PyArrow Object which has to be converted.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        SingleColumnFrame

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> cudf.Index.from_arrow(pa.array(["a", "b", None]))
        StringIndex(['a' 'b' None], dtype='object')
        >>> cudf.Series.from_arrow(pa.array(["a", "b", None]))
        0       a
        1       b
        2    <NA>
        dtype: object
        """
        return cls(ColumnBase.from_arrow(array))

    def to_arrow(self):
        """
        Convert to a PyArrow Array.

        Returns
        -------
        PyArrow Array

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["a", "b", None])
        >>> sr.to_arrow()
        <pyarrow.lib.StringArray object at 0x7f796b0e7600>
        [
          "a",
          "b",
          null
        ]
        >>> ind = cudf.Index(["a", "b", None])
        >>> ind.to_arrow()
        <pyarrow.lib.StringArray object at 0x7f796b0e7750>
        [
          "a",
          "b",
          null
        ]
        """
        return self._column.to_arrow()

    @property
    def is_unique(self):
        """Return boolean if values in the object are unique.

        Returns
        -------
        bool
        """
        return self._column.is_unique

    @property
    def is_monotonic(self):
        """Return boolean if values in the object are monotonically increasing.

        This property is an alias for :attr:`is_monotonic_increasing`.

        Returns
        -------
        bool
        """
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_decreasing

    @property
    def __cuda_array_interface__(self):
        return self._column.__cuda_array_interface__

    def factorize(self, na_sentinel=-1):
        """Encode the input values as integer labels.

        Parameters
        ----------
        na_sentinel : number
            Value to indicate missing category.

        Returns
        --------
        (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)
            - *labels* contains the encoded values
            - *cats* contains the categories in order that the N-th
              item corresponds to the (N-1) code.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'a', 'c'])
        >>> codes, uniques = s.factorize()
        >>> codes
        array([0, 0, 1], dtype=int8)
        >>> uniques
        StringIndex(['a' 'c'], dtype='object')
        """
        return cudf.core.algorithms.factorize(self, na_sentinel=na_sentinel)

    def _make_operands_for_binop(
        self,
        other: T,
        fill_value: Any = None,
        reflect: bool = False,
        *args,
        **kwargs,
    ) -> Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]]:
        """Generate the dictionary of operands used for a binary operation.

        Parameters
        ----------
        other : SingleColumnFrame
            The second operand.
        fill_value : Any, default None
            The value to replace null values with. If ``None``, nulls are not
            filled before the operation.
        reflect : bool, default False
            If ``True``, swap the order of the operands. See
            https://docs.python.org/3/reference/datamodel.html#object.__ror__
            for more information on when this is necessary.

        Returns
        -------
        Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]]
            The operands to be passed to _colwise_binop.
        """
        # Get the appropriate name for output operations involving two objects
        # that are Series-like objects. The output shares the lhs's name unless
        # the rhs is a _differently_ named Series-like object.
        if (
            isinstance(other, (SingleColumnFrame, pd.Series, pd.Index))
            and self.name != other.name
        ):
            result_name = None
        else:
            result_name = self.name

        # This needs to be tested correctly
        if isinstance(other, SingleColumnFrame):
            other = other._column
        elif not _is_scalar_or_zero_d_array(other):
            # Non-scalar right operands are valid iff they convert to columns.
            try:
                other = as_column(other)
            except Exception:
                return NotImplemented

        return {result_name: (self._column, other, reflect, fill_value)}
