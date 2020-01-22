# Copyright (c) 2018, NVIDIA CORPORATION.

import pickle
import warnings
from numbers import Number

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda, njit

import nvstrings
import rmm

import cudf
import cudf._lib as libcudf
from cudf._lib.stream_compaction import nunique as cpp_unique_count
from cudf._libxx.column import Column
from cudf.core.buffer import Buffer
from cudf.core.dtypes import CategoricalDtype
from cudf.utils import cudautils, ioutils, utils
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_scalar,
    is_string_dtype,
    np_to_pa_dtype,
)
from cudf.utils.utils import (
    buffers_from_pyarrow,
    calc_chunk_size,
    mask_bitsize,
)


class ColumnBase(Column):
    def __init__(self, data, size, dtype, mask=None, offset=0, children=()):
        """
        Parameters
        ----------
        data : Buffer
        dtype
            The type associated with the data Buffer
        mask : Buffer, optional
        children : tuple, optional
        """
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            children=children,
        )

    def __reduce__(self):
        return (
            build_column,
            (self.data, self.dtype, self.mask, self.offset, self.children),
        )

    @property
    def data_array_view(self):
        """
        View the data as a device array or nvstrings object
        """
        if self.dtype == "object":
            return self.nvstrings

        if is_categorical_dtype(self.dtype):
            return self.codes.data_array_view
        else:
            dtype = self.dtype

        result = rmm.device_array_from_ptr(
            ptr=self.data.ptr, nelem=len(self), dtype=dtype
        )
        result.gpu_data._obj = self
        return result

    @property
    def mask_array_view(self):
        """
        View the mask as a device array
        """
        result = rmm.device_array_from_ptr(
            ptr=self.mask.ptr,
            nelem=calc_chunk_size(len(self), mask_bitsize),
            dtype=np.int8,
        )
        result.gpu_data._obj = self
        return result

    def __len__(self):
        return self.size

    def to_pandas(self):
        arr = self.data_array_view
        sr = pd.Series(arr.copy_to_host())

        if self.nullable:
            mask_bits = self.mask_array_view.copy_to_host()
            mask_bytes = (
                cudautils.expand_mask_bits(len(self), mask_bits)
                .copy_to_host()
                .astype(bool)
            )
            sr[~mask_bytes] = None
        return sr

    def equals(self, other):
        if self is other:
            return True
        if other is None or len(self) != len(other):
            return False
        if len(self) == 1:
            val = self[0] == other[0]
            # when self is multiindex we need to checkall
            if isinstance(val, np.ndarray):
                return val.all()
            return bool(val)
        return self.unordered_compare("eq", other).min()

    def __sizeof__(self):
        n = self.data.size
        if self.nullable:
            n += self.mask.size
        return n

    def set_mask(self, mask):
        """
        Return a Column with the same data but new mask.

        Parameters
        ----------
        mask : 1D array-like
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        """
        if mask is None:
            return self
        mask = Buffer(mask)
        return build_column(
            self.data,
            self.dtype,
            mask=mask,
            offset=self.offset,
            children=self.children,
        )

    @staticmethod
    def from_mem_views(data_mem, mask_mem=None, null_count=None):
        """Create a Column object from a data device array (or nvstrings
           object), and an optional mask device array
        """
        raise NotImplementedError

    @classmethod
    def _concat(cls, objs, dtype=None):
        from cudf.core.series import Series
        from cudf.core.column import (
            StringColumn,
            CategoricalColumn,
            NumericalColumn,
        )

        if len(objs) == 0:
            dtype = pd.api.types.pandas_dtype(dtype)
            if is_categorical_dtype(dtype):
                dtype = CategoricalDtype()
            return column_empty(0, dtype=dtype, masked=True)

        # If all columns are `NumericalColumn` with different dtypes,
        # we cast them to a common dtype.
        # Notice, we can always cast pure null columns
        not_null_cols = list(filter(lambda o: len(o) != o.null_count, objs))
        if len(not_null_cols) > 0 and (
            len(
                [
                    o
                    for o in not_null_cols
                    if not isinstance(o, NumericalColumn)
                    or np.issubdtype(o.dtype, np.datetime64)
                ]
            )
            == 0
        ):
            col_dtypes = [o.dtype for o in not_null_cols]
            # Use NumPy to find a common dtype
            common_dtype = np.find_common_type(col_dtypes, [])
            # Cast all columns to the common dtype
            for i in range(len(objs)):
                objs[i] = objs[i].astype(common_dtype)

        # Find the first non-null column:
        head = objs[0]
        for i, obj in enumerate(objs):
            if len(obj) != obj.null_count:
                head = obj
                break

        for i, obj in enumerate(objs):
            # Check that all columns are the same type:
            if not pd.api.types.is_dtype_equal(objs[i].dtype, head.dtype):
                # if all null, cast to appropriate dtype
                if len(obj) == obj.null_count:
                    from cudf.core.column import column_empty_like

                    objs[i] = column_empty_like(
                        head, dtype=head.dtype, masked=True, newsize=len(obj)
                    )

        # Handle categories for categoricals
        if all(isinstance(o, CategoricalColumn) for o in objs):
            cats = (
                Series(ColumnBase._concat([o.categories for o in objs]))
                .drop_duplicates()
                ._column
            )
            objs = [
                o.cat()._set_categories(cats, is_unique=True) for o in objs
            ]

        head = objs[0]
        for obj in objs:
            if not (obj.dtype == head.dtype):
                raise ValueError("All series must be of same type")

        # Handle strings separately
        if all(isinstance(o, StringColumn) for o in objs):
            objs = [o.nvstrings for o in objs]
            return as_column(nvstrings.from_strings(*objs))

        # Filter out inputs that have 0 length
        objs = [o for o in objs if len(o) > 0]
        nulls = any(col.nullable for col in objs)
        newsize = sum(map(len, objs))

        if is_categorical_dtype(head):
            data = None
            data_dtype = head.codes.dtype
            children = (
                column_empty(newsize, dtype=head.codes.dtype, masked=True),
            )
        else:
            data_dtype = head.dtype
            mem = rmm.DeviceBuffer(size=newsize * data_dtype.itemsize)
            data = Buffer(mem)
            children = None

        # Allocate output mask only if there's nulls in the input objects
        mask = None
        if nulls:
            mask = Buffer(utils.make_mask(newsize))

        col = build_column(
            data=data, dtype=head.dtype, mask=mask, children=children
        )

        # Performance the actual concatenation
        if newsize > 0:
            col = libcudf.concat._column_concat(objs, col)

        return col

    def dropna(self):
        dropped_col = libcudf.stream_compaction.drop_nulls([self])
        if not dropped_col:
            return column_empty_like(self, newsize=0)
        else:
            dropped_col = dropped_col[0]
            dropped_col.mask = None
            return dropped_col

    def _get_mask_as_column(self):
        data = Buffer(cudautils.ones(len(self), dtype=np.bool_))
        mask = as_column(data=data)
        if self.nullable:
            mask = mask.set_mask(self._mask).fillna(False)
        return mask

    def _memory_usage(self, **kwargs):
        return self.__sizeof__()

    def allocate_mask(self, all_valid=True):
        """Return a new Column with a newly allocated mask buffer.
        If ``all_valid`` is True, the new mask is set to all valid.
        If ``all_valid`` is False, the new mask is set to all null.
        """
        nelem = len(self)
        mask_sz = utils.calc_chunk_size(nelem, utils.mask_bitsize)
        mask = rmm.device_array(mask_sz, dtype=utils.mask_dtype)
        if nelem > 0:
            cudautils.fill_value(mask, 0xFF if all_valid else 0)
        mask = Buffer(mask)
        return self.set_mask(mask=mask)

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : scalar, 'pandas', or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        if fillna:
            return self.fillna(self.default_na_value()).data_array_view
        else:
            return self.dropna().data_array_view

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : scalar, 'pandas', or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self.to_gpu_array(fillna=fillna).copy_to_host()

    @property
    def valid_count(self):
        """Number of non-null values"""
        return len(self) - self.null_count

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        if self.nullable:
            return self.mask_array_view
        else:
            raise ValueError("Column has no null mask")

    def copy_data(self):
        """Copy the column with a new allocation of the data but not the mask,
        which is shared by the new column.
        """
        return self.replace(data=self.data.copy())

    def copy(self, deep=True):
        """Columns are immutable, so a deep copy produces a copy of the
        underlying data and mask and a shallow copy creates a new column and
        copies the references of the data and mask.
        """
        if deep:
            return libcudf.copying.copy_column(self)
        else:
            return build_column(
                self.data,
                self.dtype,
                mask=self.mask,
                offset=self.offset,
                children=self.children,
            )

    def view(self, newcls, **kwargs):
        """View the underlying column data differently using a subclass of
        ColumnBase

        Parameters
        ----------
        newcls : ColumnBase
            The logical view to be used
        **kwargs :
            Additional paramters for instantiating instance of *newcls*.
            Valid keywords are valid parameters for ``newcls.__init__``.
            Any omitted keywords will be defaulted to the corresponding
            attributes in ``self``.
        """
        params = Column._replace_defaults(self)
        params.update(kwargs)
        if "mask" in kwargs and "null_count" not in kwargs:
            del params["null_count"]
        return newcls(**params)

    def element_indexing(self, index):
        """Default implementation for indexing to an element

        Raises
        ------
        ``IndexError`` if out-of-bound
        """
        index = np.int32(index)
        if index < 0:
            index = len(self) + index
        if index > len(self) - 1:
            raise IndexError
        val = self.data_array_view[index]  # this can raise IndexError
        if isinstance(val, nvstrings.nvstrings):
            val = val.to_host()[0]
        valid = (
            cudautils.mask_get.py_func(self.mask_array_view, index)
            if self.mask
            else True
        )
        return val if valid else None

    def __getitem__(self, arg):
        from cudf.core.column import column

        if isinstance(arg, Number):
            arg = int(arg)
            return self.element_indexing(arg)
        elif isinstance(arg, slice):

            if is_categorical_dtype(self):
                codes = self.codes[arg]
                return build_column(
                    data=None,
                    dtype=self.dtype,
                    mask=codes.mask,
                    children=(codes,),
                )

            start, stop, stride = arg.indices(len(self))
            if start == stop:
                return column_empty(0, self.dtype, masked=True)
            # compute mask slice
            if self.has_nulls:
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError(arg)

                # slicing data
                slice_data = self.data_array_view[arg]
                # slicing mask
                data_size = self.size
                bytemask = cudautils.expand_mask_bits(
                    data_size, self.mask_array_view
                )
                slice_mask = cudautils.compact_mask_bytes(bytemask[arg])
            else:
                slice_data = self.data_array_view[arg]
                slice_mask = None
            if self.dtype == "object":
                return as_column(slice_data)
            else:
                if arg.step is not None and arg.step != 1:
                    slice_data = cudautils.as_contiguous(slice_data)
                    slice_data = Buffer(slice_data)
                else:
                    # data Buffer lifetime is tied to self:
                    slice_data = Buffer(
                        data=slice_data.device_ctypes_pointer.value,
                        size=slice_data.nbytes,
                        owner=self,
                    )

                # mask Buffer lifetime is not:
                if slice_mask is not None:
                    slice_mask = Buffer(slice_mask)

                return build_column(slice_data, self.dtype, mask=slice_mask)
        else:
            arg = column.as_column(arg)
            if len(arg) == 0:
                arg = column.as_column([], dtype="int32")
            if pd.api.types.is_integer_dtype(arg.dtype):
                return self.take(arg)
            if pd.api.types.is_bool_dtype(arg.dtype):
                return self.apply_boolean_mask(arg)
            raise NotImplementedError(type(arg))

    def __setitem__(self, key, value):
        """
        Set the value of self[key] to value.

        If value and self are of different types,
        value is coerced to self.dtype
        """
        from cudf.core import column

        if isinstance(key, slice):
            key_start, key_stop, key_stride = key.indices(len(self))
            if key_stride != 1:
                raise NotImplementedError("Stride not supported in slice")
            nelem = abs(key_stop - key_start)
        else:
            key = column.as_column(key)
            if pd.api.types.is_bool_dtype(key.dtype):
                if not len(key) == len(self):
                    raise ValueError(
                        "Boolean mask must be of same length as column"
                    )
                key = column.as_column(cudautils.arange(len(self)))[key]
            nelem = len(key)

        if is_scalar(value):
            if is_categorical_dtype(self.dtype):
                from cudf.utils.cudautils import fill_value

                data = rmm.device_array(nelem, dtype=self.codes.dtype)
                fill_value(data, self._encode(value))
                value = build_categorical_column(
                    categories=self.dtype.categories,
                    codes=as_column(data),
                    ordered=self.dtype.ordered,
                )
            elif value is None:
                value = column.column_empty(nelem, self.dtype, masked=True)
            else:
                to_dtype = pd.api.types.pandas_dtype(self.dtype)
                value = utils.scalar_broadcast_to(value, nelem, to_dtype)

        value = column.as_column(value).astype(self.dtype)

        if len(value) != nelem:
            msg = (
                f"Size mismatch: cannot set value "
                f"of size {len(value)} to indexing result of size "
                f"{nelem}"
            )
            raise ValueError(msg)

        if is_categorical_dtype(value.dtype):
            value = value.cat().set_categories(self.categories)._column
            assert self.dtype == value.dtype

        if isinstance(key, slice):
            out = libcudf.copying.copy_range(
                self, value, key_start, key_stop, 0
            )
        else:
            try:
                out = libcudf.copying.scatter(value, key, self)
            except RuntimeError as e:
                if "out of bounds" in str(e):
                    raise IndexError(
                        f"index out of bounds for column of size {len(self)}"
                    )
                raise

        self._mimic_inplace(out, inplace=True)

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.nullable:
            return self
        out = cudautils.fillna(
            data=self.data_array_view, mask=self.mask_array_view, value=value
        )
        return self.replace(data=Buffer(out), mask=None, null_count=0)

    def isnull(self):
        """Identify missing values in a Column.
        """
        return libcudf.unaryops.is_null(self)

    def isna(self):
        """Identify missing values in a Column. Alias for isnull.
        """
        return self.isnull()

    def notna(self):
        """Identify non-missing values in a Column.
        """
        return libcudf.unaryops.is_not_null(self)

    def notnull(self):
        """Identify non-missing values in a Column. Alias for notna.
        """
        return self.notna()

    def _invert(self):
        """Internal convenience function for inverting masked array

        Returns
        -------
        DeviceNDArray
           logical inverted mask
        """
        if self.has_nulls:
            raise ValueError("Column must have no nulls.")
        gpu_mask = self.data_array_view
        cudautils.invert_mask(gpu_mask, gpu_mask)
        return self.replace(data=Buffer(gpu_mask), mask=None, null_count=0)

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        # FIXME: Inefficient find in CPU code
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not len(indices):
            raise ValueError("value not found")
        return indices[-1, 0]

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not len(indices):
            raise ValueError("value not found")
        return indices[-1, 0]

    def append(self, other):
        from cudf.core.column import as_column

        return ColumnBase._concat([self, as_column(other)])

    def quantile(self, q, interpolation, exact):
        if isinstance(q, Number):
            quant = [float(q)]
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            quant = q
        else:
            msg = "`q` must be either a single element, list or numpy array"
            raise TypeError(msg)
        return libcudf.quantile.quantile(self, quant, interpolation, exact)

    def take(self, indices, ignore_index=False):
        """Return Column by taking values from the corresponding *indices*.
        """
        from cudf.core.column import column_empty_like

        # Handle zero size
        if indices.size == 0:
            return column_empty_like(self, newsize=0)

        try:
            result = libcudf.copying.gather(self, indices)
        except RuntimeError as e:
            if "out of bounds" in str(e):
                raise IndexError(
                    f"index out of bounds for column of size {len(self)}"
                )
            raise

        return result

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """

        if self.has_nulls:
            raise ValueError("Column must have no nulls.")

        return cudautils.compact_mask_bytes(self.data_array_view)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @property
    def is_unique(self):
        return self.unique_count() == len(self)

    @property
    def is_monotonic(self):
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        raise (NotImplementedError)

    @property
    def is_monotonic_decreasing(self):
        raise (NotImplementedError)

    def get_slice_bound(self, label, side, kind):
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.
        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}
        """
        assert kind in ["ix", "loc", "getitem", None]
        if side not in ("left", "right"):
            raise ValueError(
                "Invalid value for side kwarg,"
                " must be either 'left' or 'right': %s" % (side,)
            )

        # TODO: Handle errors/missing keys correctly
        #       Not currently using `kind` argument.
        if side == "left":
            return self.find_first_value(label, closest=True)
        if side == "right":
            return self.find_last_value(label, closest=True) + 1

    def sort_by_values(self):
        raise NotImplementedError

    def _unique_segments(self):
        """ Common code for unique, unique_count and value_counts"""
        # make dense column
        densecol = self.dropna()
        # sort the column
        sortcol, _ = densecol.sort_by_values()
        # find segments
        sortedvals = sortcol.data_array_view
        segs, begins = cudautils.find_segments(sortedvals)
        return segs, sortedvals

    def unique_count(self, method="sort", dropna=True):
        if method != "sort":
            msg = "non sort based unique_count() not implemented yet"
            raise NotImplementedError(msg)
        return cpp_unique_count(self, dropna)

    def repeat(self, repeats, axis=None):
        assert axis in (None, 0)
        return libcudf.filling.repeat([self], repeats)[0]

    def _mimic_inplace(self, result, inplace=False):
        """
        If `inplace=True`, used to mimic an inplace operation
        by replacing data in ``self`` with data in ``result``.

        Otherwise, returns ``result`` unchanged.
        """
        if inplace:
            self.data = result.data
            self.mask = result.mask
            self.dtype = result.dtype
            self.size = result.size
            self.offset = result.offset
            if hasattr(result, "children"):
                self.children = result.children
                if is_string_dtype(self):
                    # force recomputation of nvstrings/nvcategory
                    self._nvstrings = None
                    self._nvcategory = None
            self.__class__ = result.__class__
        else:
            return result

    def astype(self, dtype, **kwargs):
        if is_categorical_dtype(dtype):
            return self.as_categorical_column(dtype, **kwargs)
        elif pd.api.types.pandas_dtype(dtype).type in (np.str_, np.object_):
            return self.as_string_column(dtype, **kwargs)

        elif np.issubdtype(dtype, np.datetime64):
            return self.as_datetime_column(dtype, **kwargs)

        else:
            return self.as_numerical_column(dtype, **kwargs)

    def as_categorical_column(self, dtype, **kwargs):
        if "ordered" in kwargs:
            ordered = kwargs["ordered"]
        else:
            ordered = False

        sr = cudf.Series(self)
        labels, cats = sr.factorize()

        # string columns include null index in factorization; remove:
        if (
            pd.api.types.pandas_dtype(self.dtype).type in (np.str_, np.object_)
        ) and self.has_nulls:
            cats = cats.dropna()
            labels = labels - 1

        return build_categorical_column(
            categories=cats,
            codes=labels._column,
            mask=self.mask,
            ordered=ordered,
        )

    def as_numerical_column(self, dtype, **kwargs):
        raise NotImplementedError

    def as_datetime_column(self, dtype, **kwargs):
        raise NotImplementedError

    def as_string_column(self, dtype, **kwargs):
        raise NotImplementedError

    def apply_boolean_mask(self, mask):
        mask = as_column(mask, dtype="bool")
        data = libcudf.stream_compaction.apply_boolean_mask([self], mask)
        if not data:
            return column_empty_like(self, newsize=0)
        else:
            result = data[0]
            return result

    def argsort(self, ascending):
        _, inds = self.sort_by_values(ascending=ascending)
        return inds

    @property
    def __cuda_array_interface__(self):
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data.ptr, True),
            "version": 1,
        }

        if self.nullable and self.has_nulls:
            from types import SimpleNamespace

            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = SimpleNamespace(
                __cuda_array_interface__={
                    "shape": (len(self),),
                    "typestr": "<t1",
                    "data": (self.mask.ptr, True),
                    "version": 1,
                }
            )
            output["mask"] = mask

        return output

    def serialize(self):
        header = {}
        frames = []
        header["type"] = pickle.dumps(type(self))
        header["dtype"] = self.dtype.str
        data_frames = [self.data_array_view]
        frames.extend(data_frames)

        if self.nullable:
            mask_frames = [self.mask_array_view]
        else:
            mask_frames = []
        frames.extend(mask_frames)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        dtype = header["dtype"]
        data = Buffer(frames[0])
        mask = None
        if header["frame_count"] > 1:
            mask = Buffer(frames[1])
        return build_column(data=data, dtype=dtype, mask=mask)


def column_empty_like(column, dtype=None, masked=False, newsize=None):
    """Allocate a new column like the given *column*
    """
    if dtype is None:
        dtype = column.dtype
    row_count = len(column) if newsize is None else newsize

    if (
        hasattr(column, "dtype")
        and is_categorical_dtype(column.dtype)
        and dtype == column.dtype
    ):
        codes = column_empty_like(column.codes, masked=masked, newsize=newsize)
        return build_column(
            data=None, dtype=dtype, mask=codes.mask, children=(codes,)
        )

    return column_empty(row_count, dtype, masked)


def column_empty_like_same_mask(column, dtype):
    """Create a new empty Column with the same length and the same mask.

    Parameters
    ----------
    dtype : np.dtype like
        The dtype of the data buffer.
    """
    result = column_empty_like(column, dtype)
    if column.nullable:
        result.mask = column.mask
    return result


def column_empty(row_count, dtype="object", masked=False):
    """Allocate a new column like the given row_count and dtype.
    """
    dtype = pd.api.types.pandas_dtype(dtype)
    children = ()

    if is_categorical_dtype(dtype):
        data = None
        children = (
            build_column(
                data=Buffer.empty(row_count * np.dtype("int32").itemsize),
                dtype="int32",
            ),
        )
    elif dtype.kind in "OU":
        data = None
        children = (
            build_column(
                data=Buffer.empty(
                    (row_count + 1) * np.dtype("int32").itemsize
                ),
                dtype="int32",
            ),
            build_column(
                data=Buffer.empty(row_count * np.dtype("int8").itemsize),
                dtype="int8",
            ),
        )
    else:
        data = Buffer.empty(row_count * dtype.itemsize)

    if masked:
        mask = Buffer(cudautils.make_empty_mask(row_count))
    else:
        mask = None

    return build_column(data, dtype, mask=mask, children=children)


def build_column(
    data, dtype, mask=None, offset=0, children=(), categories=None
):
    """
    Build a Column of the appropriate type from the given parameters

    Parameters
    ----------
    data : Buffer
        The data buffer (can be None if constructin certain Column
        types like StringColumn or CategoricalColumn)
    dtype
        The dtype associated with the Column to construct
    mask : Buffer, optionapl
        The mask buffer
    offset : int, optional
    children : tuple, optional
    categories : Column, optional
        If constructing a CategoricalColumn, a Column containing
        the categories
    """
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.categorical import CategoricalColumn
    from cudf.core.column.string import StringColumn

    dtype = pd.api.types.pandas_dtype(dtype)

    if is_categorical_dtype(dtype):
        if not len(children) == 1:
            raise ValueError(
                "Must specify exactly one child column for CategoricalColumn"
            )
        if not isinstance(children[0], ColumnBase):
            raise TypeError("children must be a tuple of Columns")
        return CategoricalColumn(
            dtype=dtype, mask=mask, offset=offset, children=children
        )
    elif dtype.type is np.datetime64:
        return DatetimeColumn(data=data, dtype=dtype, mask=mask, offset=offset)
    elif dtype.type in (np.object_, np.str_):
        return StringColumn(mask=mask, offset=offset, children=children)
    else:
        return NumericalColumn(
            data=data, dtype=dtype, mask=mask, offset=offset
        )


def build_categorical_column(
    categories, codes, mask=None, offset=0, ordered=None
):
    """
    Build a CategoricalColumn

    Parameters
    ----------
    categories : Column
        Column of categories
    codes : Column
        Column of codes, the size of the resulting Column will be
        the size of `codes`
    mask : Buffer
        Null mask
    offset : int
    ordered : bool
        Indicates whether the categories are ordered
    """
    dtype = CategoricalDtype(categories=as_column(categories), ordered=ordered)
    return build_column(
        data=None,
        dtype=dtype,
        mask=mask,
        offset=offset,
        children=(as_column(codes),),
    )


def as_column(arbitrary, nan_as_null=True, dtype=None, length=None):
    """Create a Column from an arbitrary object

    Parameters
    ----------
    arbitrary : object
        Object to construct the Column from. See *Notes*.
    nan_as_null : bool,optional
        If True (default), treat NaN values in arbitrary as null.
    dtype : optional
        Optionally typecast the construted Column to the given
        dtype.
    length : int, optional
        If `arbitrary` is a scalar, broadcast into a Column of
        the given length.

    Returns
    -------
    A Column of the appropriate type and size.

    Notes
    -----
    Currently support inputs are:

    * ``Column``
    * ``Series``
    * ``Index``
    * Scalars (can be broadcasted to a specified `length`)
    * Objects exposing ``__cuda_array_interface__`` (e.g., numba device arrays)
    * Objects exposing ``__array_interface__``(e.g., numpy arrays)
    * pyarrow array
    * pandas.Categorical objects
    """

    from cudf.core.column import numerical, categorical, datetime, string
    from cudf.core.series import Series
    from cudf.core.index import Index

    if isinstance(arbitrary, ColumnBase):
        if dtype is not None:
            return arbitrary.astype(dtype)
        else:
            return arbitrary

    elif isinstance(arbitrary, Series):
        data = arbitrary._column
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, Index):
        data = arbitrary._values
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, nvstrings.nvstrings):
        sbuf = Buffer.empty(arbitrary.byte_count())
        obuf = Buffer.empty(
            (arbitrary.size() + 1) * np.dtype("int32").itemsize
        )

        nbuf = None
        if arbitrary.null_count() > 0:
            mask_size = calc_chunk_size(arbitrary.size(), mask_bitsize)
            nbuf = Buffer.empty(mask_size)
            arbitrary.set_null_bitmask(nbuf.ptr, bdevmem=True)
        arbitrary.to_offsets(sbuf.ptr, obuf.ptr, None, bdevmem=True)
        children = (
            build_column(obuf, dtype="int32"),
            build_column(sbuf, dtype="int8"),
        )
        data = build_column(
            data=None, dtype="object", mask=nbuf, children=children
        )
        data._nvstrings = arbitrary

    elif isinstance(arbitrary, Buffer):
        if dtype is None:
            raise TypeError(f"dtype cannot be None if 'arbitrary' is a Buffer")
        data = build_column(arbitrary, dtype=dtype)

    elif cuda.devicearray.is_cuda_ndarray(arbitrary):
        data = as_column(Buffer(arbitrary), dtype=arbitrary.dtype)
        if (
            data.dtype in [np.float16, np.float32, np.float64]
            and arbitrary.size > 0
        ):
            if nan_as_null:
                mask = libcudf.unaryops.nans_to_nulls(data)
                data.mask = mask

        elif data.dtype.kind == "M":
            null = column_empty_like(data, masked=True, newsize=1)
            col = libcudf.replace.replace(
                as_column(Buffer(arbitrary), dtype=arbitrary.dtype),
                as_column(
                    Buffer(np.array([np.datetime64("NaT")], dtype=data.dtype)),
                    dtype=arbitrary.dtype,
                ),
                null,
            )
            data = datetime.DatetimeColumn(
                data=Buffer(arbitrary), dtype=data.dtype, mask=col.mask
            )

    elif hasattr(arbitrary, "__cuda_array_interface__"):
        desc = arbitrary.__cuda_array_interface__
        data = _data_from_cuda_array_interface_desc(arbitrary)
        mask = _mask_from_cuda_array_interface_desc(arbitrary)
        dtype = np.dtype(desc["typestr"])
        col = build_column(data, dtype=dtype, mask=mask)
        return col

    elif isinstance(arbitrary, np.ndarray):
        # CUDF assumes values are always contiguous
        if not arbitrary.flags["C_CONTIGUOUS"]:
            arbitrary = np.ascontiguousarray(arbitrary)

        if dtype is not None:
            arbitrary = arbitrary.astype(dtype)

        if arbitrary.dtype.kind == "M":
            data = datetime.DatetimeColumn.from_numpy(arbitrary)

        elif arbitrary.dtype.kind in ("O", "U"):
            data = as_column(pa.Array.from_pandas(arbitrary))
        else:
            data = as_column(rmm.to_device(arbitrary), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, pa.Array):
        if isinstance(arbitrary, pa.StringArray):
            nbuf, obuf, sbuf = buffers_from_pyarrow(arbitrary)
            children = (
                build_column(data=obuf, dtype="int32"),
                build_column(data=sbuf, dtype="int8"),
            )

            data = string.StringColumn(mask=nbuf, children=children)

        elif isinstance(arbitrary, pa.NullArray):
            new_dtype = pd.api.types.pandas_dtype(dtype)
            if (type(dtype) == str and dtype == "empty") or dtype is None:
                new_dtype = pd.api.types.pandas_dtype(
                    arbitrary.type.to_pandas_dtype()
                )

            if is_categorical_dtype(new_dtype):
                arbitrary = arbitrary.dictionary_encode()
            else:
                if nan_as_null:
                    arbitrary = arbitrary.cast(np_to_pa_dtype(new_dtype))
                else:
                    # casting a null array doesn't make nans valid
                    # so we create one with valid nans from scratch:
                    if new_dtype == np.dtype("object"):
                        arbitrary = utils.scalar_broadcast_to(
                            None, (len(arbitrary),), dtype=new_dtype
                        )
                    else:
                        arbitrary = utils.scalar_broadcast_to(
                            np.nan, (len(arbitrary),), dtype=new_dtype
                        )
            data = as_column(arbitrary, nan_as_null=nan_as_null)
        elif isinstance(arbitrary, pa.DictionaryArray):
            codes = as_column(arbitrary.indices)
            if isinstance(arbitrary.dictionary, pa.NullArray):
                categories = as_column([], dtype="object")
            else:
                categories = as_column(arbitrary.dictionary)
            dtype = CategoricalDtype(
                categories=categories, ordered=arbitrary.type.ordered
            )
            data = categorical.CategoricalColumn(
                dtype=dtype, mask=codes.mask, children=(codes,)
            )
        elif isinstance(arbitrary, pa.TimestampArray):
            dtype = np.dtype("M8[{}]".format(arbitrary.type.unit))
            pamask, padata, _ = buffers_from_pyarrow(arbitrary, dtype=dtype)

            data = datetime.DatetimeColumn(
                data=padata, mask=pamask, dtype=dtype
            )
        elif isinstance(arbitrary, pa.Date64Array):
            raise NotImplementedError
            pamask, padata, _ = buffers_from_pyarrow(arbitrary, dtype="M8[ms]")
            data = datetime.DatetimeColumn(
                data=padata, mask=pamask, dtype=np.dtype("M8[ms]")
            )
        elif isinstance(arbitrary, pa.Date32Array):
            # No equivalent np dtype and not yet supported
            warnings.warn(
                "Date32 values are not yet supported so this will "
                "be typecast to a Date64 value",
                UserWarning,
            )
            data = as_column(arbitrary.cast(pa.int32())).astype("M8[ms]")
        elif isinstance(arbitrary, pa.BooleanArray):
            # Arrow uses 1 bit per value while we use int8
            dtype = np.dtype(np.bool)
            # Needed because of bug in PyArrow
            # https://issues.apache.org/jira/browse/ARROW-4766
            if len(arbitrary) > 0:
                arbitrary = arbitrary.cast(pa.int8())
            else:
                arbitrary = pa.array([], type=pa.int8())
            pamask, padata, _ = buffers_from_pyarrow(arbitrary, dtype=dtype)
            data = numerical.NumericalColumn(
                data=padata, mask=pamask, dtype=dtype
            )
        else:
            pamask, padata, _ = buffers_from_pyarrow(arbitrary)
            data = numerical.NumericalColumn(
                data=padata,
                dtype=np.dtype(arbitrary.type.to_pandas_dtype()),
                mask=pamask,
            )

    elif isinstance(arbitrary, pa.ChunkedArray):
        gpu_cols = [
            as_column(chunk, dtype=dtype) for chunk in arbitrary.chunks
        ]

        if dtype and dtype != "empty":
            new_dtype = dtype
        else:
            pa_type = arbitrary.type
            if pa.types.is_dictionary(pa_type):
                new_dtype = "category"
            else:
                new_dtype = np.dtype(pa_type.to_pandas_dtype())

        data = ColumnBase._concat(gpu_cols, dtype=new_dtype)

    elif isinstance(arbitrary, (pd.Series, pd.Categorical)):
        if is_categorical_dtype(arbitrary):
            data = as_column(pa.array(arbitrary, from_pandas=True))
        elif arbitrary.dtype == np.bool:
            # Bug in PyArrow or HDF that requires us to do this
            data = as_column(pa.array(np.array(arbitrary), from_pandas=True))
        else:
            data = as_column(pa.array(arbitrary, from_pandas=nan_as_null))

    elif isinstance(arbitrary, pd.Timestamp):
        # This will always treat NaTs as nulls since it's not technically a
        # discrete value like NaN
        data = as_column(pa.array(pd.Series([arbitrary]), from_pandas=True))

    elif np.isscalar(arbitrary) and not isinstance(arbitrary, memoryview):
        length = length or 1
        data = as_column(
            utils.scalar_broadcast_to(arbitrary, length, dtype=dtype)
        )
        if not nan_as_null:
            data = data.fillna(np.nan)

    elif isinstance(arbitrary, memoryview):
        data = as_column(
            np.asarray(arbitrary), dtype=dtype, nan_as_null=nan_as_null
        )

    else:
        try:
            data = as_column(
                memoryview(arbitrary), dtype=dtype, nan_as_null=nan_as_null
            )
        except TypeError:
            pa_type = None
            np_type = None
            try:
                if dtype is not None:
                    dtype = pd.api.types.pandas_dtype(dtype)
                    if is_categorical_dtype(dtype):
                        raise TypeError
                    else:
                        np_type = np.dtype(dtype).type
                        if np_type == np.bool_:
                            pa_type = pa.bool_()
                        else:
                            pa_type = np_to_pa_dtype(np.dtype(dtype))
                data = as_column(
                    pa.array(arbitrary, type=pa_type, from_pandas=nan_as_null),
                    dtype=dtype,
                    nan_as_null=nan_as_null,
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                if is_categorical_dtype(dtype):
                    sr = pd.Series(arbitrary, dtype="category")
                    data = as_column(sr, nan_as_null=nan_as_null)
                elif np_type == np.str_:
                    sr = pd.Series(arbitrary, dtype="str")
                    data = as_column(sr, nan_as_null=nan_as_null)
                else:
                    data = as_column(
                        np.asarray(arbitrary, dtype=np.dtype(dtype)),
                        nan_as_null=nan_as_null,
                    )
    return data


def column_applymap(udf, column, out_dtype):
    """Apply a elemenwise function to transform the values in the Column.

    Parameters
    ----------
    udf : function
        Wrapped by numba jit for call on the GPU as a device function.
    column : Column
        The source column.
    out_dtype  : numpy.dtype
        The dtype for use in the output.

    Returns
    -------
    result : rmm.device_array
    """
    core = njit(udf)
    results = rmm.device_array(shape=len(column), dtype=out_dtype)
    values = column.data_array_view
    if column.nullable:
        # For masked columns
        @cuda.jit
        def kernel_masked(values, masks, results):
            i = cuda.grid(1)
            # in range?
            if i < values.size:
                # valid?
                if utils.mask_get(masks, i):
                    # call udf
                    results[i] = core(values[i])

        masks = column.mask_array_view
        kernel_masked.forall(len(column))(values, masks, results)
    else:
        # For non-masked columns
        @cuda.jit
        def kernel_non_masked(values, results):
            i = cuda.grid(1)
            # in range?
            if i < values.size:
                # call udf
                results[i] = core(values[i])

        kernel_non_masked.forall(len(column))(values, results)

    return as_column(results)


def _data_from_cuda_array_interface_desc(obj):
    desc = obj.__cuda_array_interface__
    ptr = desc["data"][0]
    nelem = desc["shape"][0]
    dtype = np.dtype(desc["typestr"])

    data = Buffer(data=ptr, size=nelem * dtype.itemsize, owner=obj)
    return data


def _mask_from_cuda_array_interface_desc(obj):
    from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize
    from cudf.utils.cudautils import compact_mask_bytes

    desc = obj.__cuda_array_interface__
    mask = desc.get("mask", None)

    if mask is not None:
        desc = mask.__cuda_array_interface__
        ptr = desc["data"][0]
        nelem = desc["shape"][0]
        typestr = desc["typestr"]
        typecode = typestr[1]
        if typecode == "t":
            nelem = calc_chunk_size(nelem, mask_bitsize)
            mask = Buffer(
                data=ptr, size=nelem * mask_dtype.itemsize, owner=obj
            )
        elif typecode == "b":
            dtype = np.dtype(typestr)
            mask = compact_mask_bytes(
                rmm.device_array_from_ptr(
                    ptr, nelem=nelem, dtype=dtype, finalizer=None
                )
            )
            mask = Buffer(mask)
        else:
            raise NotImplementedError(
                f"Cannot infer mask from typestr {typestr}"
            )
    return mask


def serialize_columns(columns):
    """
    Return the headers and frames resulting
    from serializing a list of Column
    Parameters
    ----------
    columns : list
        list of Columns to serialize
    Returns
    -------
    headers : list
        list of header metadata for each Column
    frames : list
        list of frames
    """
    headers = []
    frames = []

    if len(columns) > 0:
        header_columns = [c.serialize() for c in columns]
        headers, column_frames = zip(*header_columns)
        for f in column_frames:
            frames.extend(f)

    return headers, frames


def deserialize_columns(headers, frames):
    """
    Construct a list of Columns from a list of headers
    and frames.
    """
    columns = []

    for meta in headers:
        col_frame_count = meta["frame_count"]
        col_typ = pickle.loads(meta["type"])
        colobj = col_typ.deserialize(meta, frames[:col_frame_count])
        columns.append(colobj)
        # Advance frames
        frames = frames[col_frame_count:]

    return columns
