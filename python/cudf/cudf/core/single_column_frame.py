# Copyright (c) 2021-2024, NVIDIA CORPORATION.
"""Base class for Frame types that only have a single column."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Self

import cudf
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_integer,
    is_numeric_dtype,
)
from cudf.core.column import ColumnBase, as_column
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import NotIterable

if TYPE_CHECKING:
    from collections.abc import Hashable

    import cupy
    import numpy
    import pyarrow as pa

    from cudf._typing import NotImplementedType, ScalarLike


class SingleColumnFrame(Frame, NotIterable):
    """A one-dimensional frame.

    Frames with only a single column (Index or Series)
    share certain logic that is encoded in this class.
    """

    _SUPPORT_AXIS_LOOKUP = {
        0: 0,
        "index": 0,
    }

    @_performance_tracking
    def _reduce(
        self,
        op,
        axis=no_default,
        numeric_only=False,
        **kwargs,
    ):
        if axis not in (None, 0, no_default):
            raise NotImplementedError("axis parameter is not implemented yet")

        if numeric_only and not is_numeric_dtype(self.dtype):
            raise TypeError(
                f"Series.{op} does not allow numeric_only={numeric_only} "
                "with non-numeric dtypes."
            )
        try:
            return getattr(self._column, op)(**kwargs)
        except AttributeError:
            raise TypeError(f"cannot perform {op} with type {self.dtype}")

    @_performance_tracking
    def _scan(self, op, axis=None, *args, **kwargs):
        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        return super()._scan(op, axis=axis, *args, **kwargs)

    @property  # type: ignore
    @_performance_tracking
    def name(self):
        """Get the name of this object."""
        return next(iter(self._column_names))

    @name.setter  # type: ignore
    @_performance_tracking
    def name(self, value):
        self._data[value] = self._data.pop(self.name)

    @property  # type: ignore
    @_performance_tracking
    def ndim(self) -> int:
        """Number of dimensions of the underlying data, by definition 1."""
        return 1

    @property  # type: ignore
    @_performance_tracking
    def shape(self) -> tuple[int]:
        """Get a tuple representing the dimensionality of the Index."""
        return (len(self),)

    @property  # type: ignore
    @_performance_tracking
    def _num_columns(self) -> int:
        return 1

    @property  # type: ignore
    @_performance_tracking
    def _column(self) -> ColumnBase:
        return next(iter(self._columns))

    @property  # type: ignore
    @_performance_tracking
    def values(self) -> cupy.ndarray:
        return self._column.values

    @property  # type: ignore
    @_performance_tracking
    def values_host(self) -> numpy.ndarray:
        return self._column.values_host

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None
    ) -> Self:
        """Constructor for a single Column."""
        ca = ColumnAccessor({name: column}, verify=False)
        return cls._from_data(ca)

    @classmethod
    @_performance_tracking
    def from_arrow(cls, array) -> Self:
        raise NotImplementedError

    @_performance_tracking
    def to_arrow(self) -> pa.Array:
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

    def _to_frame(
        self, name: Hashable, index: cudf.Index | None
    ) -> cudf.DataFrame:
        """Helper function for Series.to_frame, Index.to_frame"""
        if name is no_default:
            col_name = 0 if self.name is None else self.name
        else:
            col_name = name
        ca = ColumnAccessor({col_name: self._column}, verify=False)
        return cudf.DataFrame._from_data(ca, index=index)

    @property  # type: ignore
    @_performance_tracking
    def is_unique(self) -> bool:
        """Return boolean if values in the object are unique.

        Returns
        -------
        bool
        """
        return self._column.is_unique

    @property  # type: ignore
    @_performance_tracking
    def is_monotonic_increasing(self) -> bool:
        """Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_increasing

    @property  # type: ignore
    @_performance_tracking
    def is_monotonic_decreasing(self) -> bool:
        """Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_decreasing

    @property  # type: ignore
    @_performance_tracking
    def __cuda_array_interface__(self):
        # While the parent column class has a `__cuda_array_interface__` method
        # defined, it is not implemented for all column types. When it is not
        # implemented, though, at the Frame level we really want to throw an
        # AttributeError.
        try:
            return self._column.__cuda_array_interface__
        except NotImplementedError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                "'__cuda_array_interface__'"
            )

    @_performance_tracking
    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[cupy.ndarray, cudf.Index]:
        """Encode the input values as integer labels.

        Parameters
        ----------
        sort : bool, default True
            Sort uniques and shuffle codes to maintain the relationship.
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NA values.
            If False, NA values will be encoded as non-negative
            integers and will not drop the NA from the uniques
            of the values.

        Returns
        -------
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
        Index(['a', 'c'], dtype='object')
        """
        return cudf.core.algorithms.factorize(
            self,
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )

    @_performance_tracking
    def _make_operands_for_binop(
        self,
        other: Any,
        fill_value: Any = None,
        reflect: bool = False,
    ) -> (
        dict[str | None, tuple[ColumnBase, Any, bool, Any]]
        | NotImplementedType
    ):
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
        if isinstance(
            other, SingleColumnFrame
        ) and not cudf.utils.utils._is_same_name(self.name, other.name):
            result_name = None
        else:
            result_name = self.name

        if isinstance(other, SingleColumnFrame):
            other = other._column
        elif not _is_scalar_or_zero_d_array(other):
            if not hasattr(
                other, "__cuda_array_interface__"
            ) and not isinstance(other, cudf.RangeIndex):
                return NotImplemented

            # Non-scalar right operands are valid iff they convert to columns.
            try:
                other = as_column(other)
            except Exception:
                return NotImplemented

        return {result_name: (self._column, other, reflect, fill_value)}

    @_performance_tracking
    def nunique(self, dropna: bool = True) -> int:
        """
        Return count of unique values for the column.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        int
            Number of unique values in the column.
        """
        return self._column.distinct_count(dropna=dropna)

    def _get_elements_from_column(self, arg) -> ScalarLike | ColumnBase:
        # A generic method for getting elements from a column that supports a
        # wide range of different inputs. This method should only used where
        # _absolutely_ necessary, since in almost all cases a more specific
        # method can be used e.g. element_indexing or slice.
        if _is_scalar_or_zero_d_array(arg):
            if not is_integer(arg):
                raise ValueError(
                    "Can only select elements with an integer, "
                    f"not a {type(arg).__name__}"
                )
            return self._column.element_indexing(int(arg))
        elif isinstance(arg, slice):
            start, stop, stride = arg.indices(len(self))
            return self._column.slice(start, stop, stride)
        else:
            arg = as_column(arg)
            if len(arg) == 0:
                arg = cudf.core.column.column_empty(0, dtype="int32")
            if arg.dtype.kind in "iu":
                return self._column.take(arg)
            if arg.dtype.kind == "b":
                if (bn := len(arg)) != (n := len(self)):
                    raise IndexError(
                        f"Boolean mask has wrong length: {bn} not {n}"
                    )
                return self._column.apply_boolean_mask(arg)
            raise NotImplementedError(f"Unknown indexer {type(arg)}")

    @_performance_tracking
    def where(self, cond, other=None, inplace=False):
        from cudf.core._internals.where import (
            _check_and_cast_columns_with_other,
        )

        if isinstance(other, cudf.DataFrame):
            raise NotImplementedError(
                "cannot align with a higher dimensional Frame"
            )
        cond = as_column(cond)
        if len(cond) != len(self):
            raise ValueError(
                """Array conditional must be same shape as self"""
            )

        if not cudf.api.types.is_scalar(other):
            other = cudf.core.column.as_column(other)

        input_col, other = _check_and_cast_columns_with_other(
            source_col=self._column, other=other, inplace=inplace
        )

        result = cudf._lib.copying.copy_if_else(input_col, other, cond)
        return result._with_type_metadata(self.dtype)

    @_performance_tracking
    def transpose(self):
        """Return the transpose, which is by definition self."""
        return self

    T = property(transpose, doc=transpose.__doc__)
