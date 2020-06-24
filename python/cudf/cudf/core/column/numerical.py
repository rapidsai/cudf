# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_integer_dtype

import cudf
import cudf._lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar
from cudf.core.buffer import Buffer
from cudf.core.column import as_column, column
from cudf.utils import cudautils, utils
from cudf.utils.dtypes import (
    min_column_type,
    min_signed_type,
    np_to_pa_dtype,
    numeric_normalize_types,
)


class NumericalColumn(column.ColumnBase):
    def __init__(
        self, data, dtype, mask=None, size=None, offset=0, null_count=None
    ):
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
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
        )

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
        # TODO: Use `scalar`-based `contains` wrapper
        return libcudf.search.contains(
            self, column.as_column([item], dtype=self.dtype)
        ).any()

    def unary_operator(self, unaryop):
        return _numeric_column_unaryop(self, op=unaryop)

    def binary_operator(self, binop, rhs, reflect=False):
        int_dtypes = [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
        ]
        tmp = rhs
        if reflect:
            tmp = self
        if isinstance(rhs, (NumericalColumn, Scalar)) or np.isscalar(rhs):
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
            lhs=self, rhs=rhs, op=binop, out_dtype=out_dtype, reflect=reflect
        )

    def _apply_scan_op(self, op):
        return libcudf.reduce.scan(op, self, True)

    def normalize_binop_value(self, other):
        if other is None:
            return other
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in {"b", "i", "u", "f"}:
            other_dtype = np.promote_types(self.dtype, other_dtype)
            if other_dtype == np.dtype("float16"):
                other = np.dtype("float32").type(other)
                other_dtype = other.dtype
            if self.dtype.kind == "b":
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

    def int2ip(self):
        if self.dtype != np.dtype("int64"):
            raise TypeError("Only int64 type can be converted to ip")

        return libcudf.string_casting.int2ip(self)

    def as_string_column(self, dtype, **kwargs):
        from cudf.core.column import string, as_column

        if len(self) > 0:
            return string._numeric_to_str_typecast_functions[
                np.dtype(self.dtype)
            ](self, **kwargs)
        else:
            return as_column([], dtype="object")

    def as_datetime_column(self, dtype, **kwargs):
        from cudf.core.column import build_column

        return build_column(
            data=self.astype("int64").base_data,
            dtype=dtype,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )

    def as_numerical_column(self, dtype, **kwargs):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype)

    def to_pandas(self, index=None):
        if self.has_nulls and self.dtype == np.bool:
            # Boolean series in Pandas that contains None/NaN is of dtype
            # `np.object`, which is not natively supported in GDF.
            ret = self.astype(np.int8).fillna(-1).to_array()
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

        data = Buffer(
            cudautils.apply_round(self.data_array_view, decimals).view("|u1")
        )
        return column.build_column(data=data, dtype=self.dtype, mask=self.mask)

    def applymap(self, udf, out_dtype=None):
        """Apply an element-wise function to transform the values in the Column.

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
        elif dkind == "i":
            return np.iinfo(self.dtype).min
        elif dkind == "u":
            return np.iinfo(self.dtype).max
        elif dkind == "b":
            return self.dtype.type(False)
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
        return libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
        )

    def fillna(self, fill_value):
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
        result = column.build_column(
            result.base_data,
            result.dtype,
            mask=None,
            offset=result.offset,
            size=result.size,
        )

        return result

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
        elif to_dtype.kind == "f" and self.dtype.kind in {"i", "u"}:
            info = np.finfo(to_dtype)
            biggest_exact_int = 2 ** (info.nmant + 1)
            if (self.min() >= -biggest_exact_int) and (
                self.max() <= biggest_exact_int
            ):
                return True
            else:

                filled = self.fillna(0)
                if (
                    cudf.Series(filled).astype(to_dtype).astype(filled.dtype)
                    == cudf.Series(filled)
                ).all():
                    return True
                else:
                    return False

        # want to cast float to int:
        elif to_dtype.kind in {"i", "u"} and self.dtype.kind == "f":
            info = np.iinfo(to_dtype)
            min_, max_ = info.min, info.max
            # best we can do is hope to catch it here and avoid compare
            if (self.min() >= min_) and (self.max() <= max_):

                filled = self.fillna(0)
                if (cudf.Series(filled) % 1 == 0).all():
                    return True
                else:
                    return False
            else:
                return False


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def _numeric_column_binop(lhs, rhs, op, out_dtype, reflect=False):
    if reflect:
        lhs, rhs = rhs, lhs

    is_op_comparison = op in ["lt", "gt", "le", "ge", "eq", "ne"]

    if is_op_comparison:
        out_dtype = "bool"

    out = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)

    if is_op_comparison:
        out = out.fillna(op == "ne")

    return out


def _numeric_column_unaryop(operand, op):
    if callable(op):
        return libcudf.transform.transform(operand, op)

    op = libcudf.unary.UnaryOp[op.upper()]
    return libcudf.unary.unary_operation(operand, op)


def _safe_cast_to_int(col, dtype):
    """
    Cast given NumericalColumn to given integer dtype safely.
    """
    assert is_integer_dtype(dtype)

    if col.dtype == dtype:
        return col

    new_col = col.astype(dtype)
    if new_col.binary_operator("eq", col).all():
        return new_col
    else:
        raise TypeError(
            "Cannot safely cast non-equivalent {} to {}".format(
                col.dtype.type.__name__, np.dtype(dtype).type.__name__
            )
        )


def _normalize_find_and_replace_input(input_column_dtype, col_to_normalize):
    normalized_column = column.as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        col_to_normalize_dtype = min_column_type(
            normalized_column, input_column_dtype
        )
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
        col_to_normalize_dtype.kind == "f"
        and input_column_dtype.kind in {"i", "u"}
    ) or (col_to_normalize_dtype.num > input_column_dtype.num):
        raise TypeError(
            f"Potentially unsafe cast for non-equivalent "
            f"{col_to_normalize_dtype.name} "
            f"to {input_column_dtype.name}"
        )
    return normalized_column.astype(input_column_dtype)


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
    bins_buf = Buffer(bins.view("|u1"))
    bin_col = NumericalColumn(data=bins_buf, dtype=bins.dtype)
    return as_column(
        libcudf.sort.digitize(column.as_frame(), bin_col.as_frame(), right)
    )
