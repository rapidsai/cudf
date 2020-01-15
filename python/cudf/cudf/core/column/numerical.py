# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from __future__ import division, print_function

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_integer_dtype

import rmm

import cudf._lib as libcudf
from cudf.core._sort import get_sorted_inds
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.utils import cudautils, utils
from cudf.utils.dtypes import (
    min_numeric_column_type,
    min_signed_type,
    np_to_pa_dtype,
    numeric_normalize_types,
)


class NumericalColumn(column.ColumnBase):
    def __init__(self, data, dtype, mask=None, offset=0):
        """
        Parameters
        ----------
        data : Buffer
        dtype : np.dtype
            The dtype associated with the data Buffer
        mask : Buffer, optional
        """
        dtype = np.dtype(dtype)
        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        size = data.size // dtype.itemsize
        super().__init__(data, size=size, dtype=dtype, mask=mask)

    def __contains__(self, item):
        """
        Returns True if column contains item, else False.
        """
        # Handles improper item types
        # Fails if item is of type None, so the handler.
        try:
            if np.can_cast(item, self.data_array_view.dtype):
                item = self.data_array_view.dtype.type(item)
            else:
                return False
        except Exception:
            return False
        return libcudf.search.contains(self, item)

    def binary_operator(self, binop, rhs, reflect=False):
        int_dtypes = [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
        ]
        tmp = rhs
        if reflect:
            tmp = self
        if isinstance(rhs, NumericalColumn) or np.isscalar(rhs):
            out_dtype = np.result_type(self.dtype, rhs.dtype)
            if binop in ["mod", "floordiv"]:
                if (tmp.dtype in int_dtypes) and (
                    (np.isscalar(tmp) and (0 == tmp))
                    or ((isinstance(tmp, NumericalColumn)) and (0.0 in tmp))
                ):
                    out_dtype = np.dtype("float_")
        elif rhs is None:
            out_dtype = self.dtype
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))
        return _numeric_column_binop(
            lhs=self, rhs=rhs, op=binop, out_dtype=out_dtype, reflect=reflect,
        )

    def unary_operator(self, unaryop):
        return _numeric_column_unaryop(self, op=unaryop)

    def unordered_compare(self, cmpop, rhs):
        return _numeric_column_compare(self, rhs, op=cmpop)

    def ordered_compare(self, cmpop, rhs):
        return _numeric_column_compare(self, rhs, op=cmpop)

    def _apply_scan_op(self, op):
        out_col = column.column_empty_like_same_mask(self, dtype=self.dtype)
        libcudf.reduce.scan(self, out_col, op, inclusive=True)
        return out_col

    def normalize_binop_value(self, other):
        if other is None:
            return other
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in "biuf":
            other_dtype = np.promote_types(self.dtype, other_dtype)
            if other_dtype == np.dtype("float16"):
                other = np.dtype("float32").type(other)
                other_dtype = other.dtype
            if other_dtype.kind in "u":
                other_dtype = min_signed_type(other)
            if np.isscalar(other):
                other = np.dtype(other_dtype).type(other)
                return other
            else:
                ary = utils.scalar_broadcast_to(
                    other, size=len(self), dtype=other_dtype
                )
                return column.build_column(
                    data=Buffer.from_array_lik(ary),
                    dtype=ary.dtype,
                    mask=self.mask,
                )
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

    def as_string_column(self, dtype, **kwargs):
        from cudf.core.column import string, as_column

        if len(self) > 0:
            if self.dtype in (np.dtype("int8"), np.dtype("int16")):
                self_as_int32 = self.astype("int32", **kwargs)
                dev_array = self_as_int32.data_array_view
                dev_ptr = self_as_int32.data.ptr
            else:
                dev_array = self.data_array_view
                dev_ptr = self.data.ptr
            null_ptr = None
            if self.nullable:
                null_ptr = self.mask.ptr
            kwargs = {"count": len(self), "nulls": null_ptr, "bdevmem": True}
            data = string._numeric_to_str_typecast_functions[
                np.dtype(dev_array.dtype)
            ](dev_ptr, **kwargs)
            return as_column(data)
        else:
            return as_column([], dtype="object")

    def as_datetime_column(self, dtype, **kwargs):
        from cudf.core.column import build_column

        return build_column(
            data=self.astype("int64").data, dtype=dtype, mask=self.mask,
        )

    def as_numerical_column(self, dtype, **kwargs):
        return column.build_column(
            data=libcudf.typecast.cast(self, dtype).data,
            dtype=np.dtype(dtype),
            mask=self.mask,
        )

    def sort_by_values(self, ascending=True, na_position="last"):
        sort_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = self[sort_inds]
        col_inds = column.build_column(
            sort_inds.data, dtype=sort_inds.dtype, mask=sort_inds.mask
        )
        return col_keys, col_inds

    def to_pandas(self, index=None):
        if self.has_nulls and self.dtype == np.bool:
            # Boolean series in Pandas that contains None/NaN is of dtype
            # `np.object`, which is not natively supported in GDF.
            ret = self.astype(np.int8).to_array(fillna=-1)
            ret = pd.Series(ret, index=index)
            ret = ret.where(ret >= 0, other=None)
            ret.replace(to_replace=1, value=True, inplace=True)
            ret.replace(to_replace=0, value=False, inplace=True)
            return ret
        else:
            return pd.Series(self.to_array(fillna="pandas"), index=index)

    def to_arrow(self):
        mask = None
        if self.nullable:
            mask = pa.py_buffer(self.mask_array_view.copy_to_host())
        data = pa.py_buffer(self.data_array_view.copy_to_host())
        pa_dtype = np_to_pa_dtype(self.dtype)
        out = pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )
        if self.dtype == np.bool:
            return out.cast(pa.bool_())
        else:
            return out

    def unique(self, method="sort"):
        # method variable will indicate what algorithm to use to
        # calculate unique, not used right now
        if method != "sort":
            msg = "non sort based unique() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out_col = column.as_column(sortedvals)[segs]
        return out_col

    def all(self):
        return bool(libcudf.reduce.reduce("all", self, dtype=np.bool_))

    def any(self):
        if self.valid_count == 0:
            return False
        return bool(libcudf.reduce.reduce("any", self, dtype=np.bool_))

    def min(self, dtype=None):
        return libcudf.reduce.reduce("min", self, dtype=dtype)

    def max(self, dtype=None):
        return libcudf.reduce.reduce("max", self, dtype=dtype)

    def sum(self, dtype=None):
        return libcudf.reduce.reduce("sum", self, dtype=dtype)

    def product(self, dtype=None):
        return libcudf.reduce.reduce("product", self, dtype=dtype)

    def mean(self, dtype=np.float64):
        return libcudf.reduce.reduce("mean", self, dtype=dtype)

    def var(self, ddof=1, dtype=np.float64):
        return libcudf.reduce.reduce("var", self, dtype=dtype, ddof=ddof)

    def std(self, ddof=1, dtype=np.float64):
        return libcudf.reduce.reduce("std", self, dtype=dtype, ddof=ddof)

    def sum_of_squares(self, dtype=None):
        return libcudf.reduce.reduce("sum_of_squares", self, dtype=dtype)

    def round(self, decimals=0):
        if decimals < 0:
            msg = "Decimal values < 0 are not yet supported."
            raise NotImplementedError(msg)

        if np.issubdtype(self.dtype, np.integer):
            return self

        data = Buffer(cudautils.apply_round(self.data_array_view, decimals))
        return column.build_column(
            data=data, dtype=self.dtype, mask=self.mask,
        )

    def applymap(self, udf, out_dtype=None):
        """Apply a elemenwise function to transform the values in the Column.

        Parameters
        ----------
        udf : function
            Wrapped by numba jit for call on the GPU as a device function.
        out_dtype  : numpy.dtype; optional
            The dtype for use in the output.
            By default, use the same dtype as *self.dtype*.

        Returns
        -------
        result : Column
            The mask is preserved.
        """
        if out_dtype is None:
            out_dtype = self.dtype
        out = column.column_applymap(udf=udf, column=self, out_dtype=out_dtype)
        return out

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == "f":
            return self.dtype.type(np.nan)
        elif dkind in "iu":
            return -1
        elif dkind == "b":
            return False
        else:
            raise TypeError(
                "numeric column of {} has no NaN value".format(self.dtype)
            )

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*.
        """
        to_replace_col = _normalize_find_and_replace_input(
            self.dtype, to_replace
        )
        if all_nan:
            replacement_col = column.as_column(replacement, dtype=self.dtype)
        else:
            replacement_col = _normalize_find_and_replace_input(
                self.dtype, replacement
            )
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = column.as_column(
                utils.scalar_broadcast_to(
                    replacement[0], (len(to_replace_col),), self.dtype
                )
            )
        replaced = self.copy()
        to_replace_col, replacement_col, replaced = numeric_normalize_types(
            to_replace_col, replacement_col, replaced
        )
        output = libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
        )
        return output

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with *fill_value*
        """
        if np.isscalar(fill_value):
            # castsafely to the same dtype as self
            fill_value_casted = self.dtype.type(fill_value)
            if not np.isnan(fill_value) and (fill_value_casted != fill_value):
                raise TypeError(
                    "Cannot safely cast non-equivalent {} to {}".format(
                        type(fill_value).__name__, self.dtype.name
                    )
                )
            fill_value = fill_value_casted
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
            # cast safely to the same dtype as self
            if is_integer_dtype(self.dtype):
                fill_value = _safe_cast_to_int(fill_value, self.dtype)
            else:
                fill_value = fill_value.astype(self.dtype)
        result = libcudf.replace.replace_nulls(self, fill_value)
        return self._mimic_inplace(result, inplace)

    def find_first_value(self, value, closest=False):
        """
        Returns offset of first value that matches. For monotonic
        columns, returns the offset of the first larger value
        if closest=True.
        """
        found = 0
        if len(self):
            found = cudautils.find_first(self.data_array_view, value)
        if found == -1 and self.is_monotonic and closest:
            if value < self.min():
                found = 0
            elif value > self.max():
                found = len(self)
            else:
                found = cudautils.find_first(
                    self.data_array_view, value, compare="gt"
                )
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

    def find_last_value(self, value, closest=False):
        """
        Returns offset of last value that matches. For monotonic
        columns, returns the offset of the last smaller value
        if closest=True.
        """
        found = 0
        if len(self):
            found = cudautils.find_last(self.data_array_view, value)
        if found == -1 and self.is_monotonic and closest:
            if value < self.min():
                found = -1
            elif value > self.max():
                found = len(self) - 1
            else:
                found = cudautils.find_last(
                    self.data_array_view, value, compare="lt"
                )
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

    def searchsorted(self, value, side="left"):
        value_col = column.as_column(value)
        return libcudf.search.search_sorted(self, value_col, side)

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_increasing = False
            else:
                self._is_monotonic_increasing = libcudf.issorted.issorted(
                    [self]
                )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_decreasing = False
            else:
                self._is_monotonic_decreasing = libcudf.issorted.issorted(
                    [self], [1]
                )
        return self._is_monotonic_decreasing

    def can_cast_safely(self, to_dtype):
        """
        Returns true if all the values in self can be
        safely cast to dtype
        """
        if self.dtype.kind == to_dtype.kind:
            if self.dtype <= to_dtype:
                return True
            else:
                # Kinds are the same but to_dtype is smaller
                if "float" in to_dtype.name:
                    info = np.finfo(to_dtype)
                elif "int" in to_dtype.name:
                    info = np.iinfo(to_dtype)
                min_, max_ = info.min, info.max

                if (self.min() > min_) and (self.max() < max_):
                    return True
                else:
                    return False

        # want to cast int to float
        elif to_dtype.kind == "f" and self.dtype.kind == "i":
            info = np.finfo(to_dtype)
            biggest_exact_int = 2 ** (info.nmant + 1)
            if (self.min() >= -biggest_exact_int) and (
                self.max() <= biggest_exact_int
            ):
                return True
            else:
                from cudf import Series

                if (
                    Series(self).astype(to_dtype).astype(self.dtype)
                    == Series(self)
                ).all():
                    return True
                else:
                    return False

        # want to cast float to int:
        elif to_dtype.kind == "i" and self.dtype.kind == "f":
            info = np.iinfo(to_dtype)
            min_, max_ = info.min, info.max
            # best we can do is hope to catch it here and avoid compare
            if (self.min() >= min_) and (self.max() <= max_):
                from cudf import Series

                if (Series(self) % 1 == 0).all():
                    return True
                else:
                    return False
            else:
                return False


def _numeric_column_binop(lhs, rhs, op, out_dtype, reflect=False):
    if reflect:
        lhs, rhs = rhs, lhs
    libcudf.nvtx.nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = False
    if np.isscalar(lhs):
        masked = rhs.nullable
        row_count = len(rhs)
    elif np.isscalar(rhs):
        masked = lhs.nullable
        row_count = len(lhs)
    elif rhs is None:
        masked = True
        row_count = len(lhs)
    elif lhs is None:
        masked = True
        row_count = len(rhs)
    else:
        masked = lhs.nullable or rhs.nullable
        row_count = len(lhs)

    is_op_comparison = op in ["lt", "gt", "le", "ge", "eq", "ne"]

    out = column.column_empty(row_count, dtype=out_dtype, masked=masked)

    _ = libcudf.binops.apply_op(lhs, rhs, out, op)

    if is_op_comparison:
        out.fillna(op == "ne", inplace=True)

    libcudf.nvtx.nvtx_range_pop()
    return out


def _numeric_column_unaryop(operand, op):
    out = libcudf.unaryops.apply_unary_op(operand, op)
    return out


def _numeric_column_compare(lhs, rhs, op):
    return _numeric_column_binop(lhs, rhs, op, out_dtype=np.bool_)


def _safe_cast_to_int(col, dtype):
    """
    Cast given NumericalColumn to given integer dtype safely.
    """
    assert is_integer_dtype(dtype)

    if col.dtype == dtype:
        return col

    new_col = col.astype(dtype)
    if new_col.unordered_compare("eq", col).all():
        return new_col
    else:
        raise TypeError(
            "Cannot safely cast non-equivalent {} to {}".format(
                col.dtype.type.__name__, np.dtype(dtype).type.__name__
            )
        )


def _normalize_find_and_replace_input(input_column_dtype, col_to_normalize):
    normalized_column = column.as_column(col_to_normalize)
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        col_to_normalize_dtype = min_numeric_column_type(normalized_column)
        # Scalar case
        if len(col_to_normalize) == 1:
            col_to_normalize_casted = input_column_dtype.type(
                col_to_normalize[0]
            )
            if not np.isnan(col_to_normalize_casted) and (
                col_to_normalize_casted != col_to_normalize[0]
            ):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{col_to_normalize[0]} "
                    f"to {input_column_dtype.name}"
                )
            else:
                col_to_normalize_dtype = input_column_dtype
    elif hasattr(col_to_normalize, "dtype"):
        col_to_normalize_dtype = col_to_normalize.dtype
    else:
        raise TypeError(f"Type {type(col_to_normalize)} not supported")

    if (
        col_to_normalize_dtype.kind == "f" and input_column_dtype.kind == "i"
    ) or (col_to_normalize_dtype > input_column_dtype):
        raise TypeError(
            f"Potentially unsafe cast for non-equivalent "
            f"{col_to_normalize_dtype.name} "
            f"to {input_column_dtype.name}"
        )
    return normalized_column.astype(input_column_dtype)


def column_hash_values(column0, *other_columns, initial_hash_values=None):
    """Hash all values in the given columns.
    Returns a new NumericalColumn[int32]
    """
    from cudf.core.column import column_empty

    columns = [column0] + list(other_columns)
    result = column_empty(len(column0), dtype=np.int32, masked=False)
    if initial_hash_values:
        initial_hash_values = rmm.to_device(initial_hash_values)
    libcudf.hash.hash_columns(columns, result, initial_hash_values)
    return result


def digitize(column, bins, right=False):
    """Return the indices of the bins to which each value in column belongs.

    Parameters
    ----------
    column : Column
        Input column.
    bins : np.array
        1-D monotonically increasing array of bins with same type as `column`.
    right : bool
        Indicates whether interval contains the right or left bin edge.

    Returns
    -------
    A device array containing the indices
    """
    assert column.dtype == bins.dtype
    bins_buf = Buffer(rmm.to_device(bins))
    bin_col = NumericalColumn(data=bins_buf, dtype=bins.dtype)
    return libcudf.sort.digitize(column, bin_col, right)
