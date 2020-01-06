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
from cudf.core.buffer import Buffer
from cudf.utils import cudautils, ioutils, utils
from cudf.utils.dtypes import is_categorical_dtype, is_scalar, np_to_pa_dtype
from cudf.utils.utils import buffers_from_pyarrow


class Column(object):
    """An immutable structure for storing data and mask for a column.
    This should be considered as the physical layer that provides
    container operations on the data and mask.

    These operations work on each data element as plain-old-data.
    Any logical operations are implemented in subclasses of *TypedColumnBase*.

    Attributes
    ----------
    _data : Buffer
        The data buffer
    _mask : Buffer
        The validity mask
    _null_count : int
        Number of null values in the mask.

    These attributes are exported in the properties (e.g. *data*, *mask*,
    *null_count*).
    """

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
            if dtype.type in (np.object_, np.str_):
                return StringColumn(data=nvstrings.to_device([]), null_count=0)
            elif is_categorical_dtype(dtype):
                return CategoricalColumn(
                    data=as_column(Buffer.null(np.dtype("int8"))),
                    null_count=0,
                    ordered=False,
                )
            else:
                return as_column(Buffer.null(dtype))

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
            if not objs[i].is_type_equivalent(head):
                # if all null, cast to appropriate dtype
                if len(obj) == obj.null_count:
                    from cudf.core.column import column_empty_like

                    objs[i] = column_empty_like(
                        head, dtype=head.dtype, masked=True, newsize=len(obj)
                    )

        # Handle categories for categoricals
        if all(isinstance(o, CategoricalColumn) for o in objs):
            cats = (
                Series(Column._concat([o.categories for o in objs]))
                .drop_duplicates()
                ._column
            )
            objs = [
                o.cat()._set_categories(cats, is_unique=True) for o in objs
            ]

        head = objs[0]
        for obj in objs:
            if not (obj.is_type_equivalent(head)):
                raise ValueError("All series must be of same type")

        # Handle strings separately
        if all(isinstance(o, StringColumn) for o in objs):
            objs = [o._data for o in objs]
            return StringColumn(
                data=nvstrings.from_strings(*objs), name=head.name
            )

        # Filter out inputs that have 0 length
        objs = [o for o in objs if len(o) > 0]
        nulls = sum(o.null_count for o in objs)
        newsize = sum(map(len, objs))
        mem = rmm.device_array(shape=newsize, dtype=head.data.dtype)
        data = Buffer.from_empty(mem, size=newsize)

        # Allocate output mask only if there's nulls in the input objects
        mask = None
        if nulls:
            mask = Buffer(utils.make_mask(newsize))

        col = head.replace(data=data, mask=mask, null_count=nulls)

        # Performance the actual concatenation
        if newsize > 0:
            col = libcudf.concat._column_concat(objs, col)

        return col

    @staticmethod
    def from_mem_views(data_mem, mask_mem=None, null_count=None, name=None):
        """Create a Column object from a data device array (or nvstrings
           object), and an optional mask device array
        """
        from cudf.core.column import column

        if isinstance(data_mem, nvstrings.nvstrings):
            return column.build_column(
                name=name,
                buffer=data_mem,
                dtype=np.dtype("object"),
                null_count=null_count,
            )
        else:
            data_buf = Buffer(data_mem)
            mask = None
            if mask_mem is not None:
                mask = Buffer(mask_mem)
            return column.build_column(
                name=name,
                buffer=data_buf,
                dtype=data_mem.dtype,
                mask=mask,
                null_count=null_count,
            )

    def __init__(self, data, mask=None, null_count=None, name=None):
        """
        Parameters
        ----------
        data : Buffer
            The code values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        """
        # Forces Column content to be contiguous
        if not data.is_contiguous():
            data = data.as_contiguous()

        assert mask is None or mask.is_contiguous()
        self._data = data
        self._mask = mask
        self._name = name

        if mask is None:
            null_count = 0
        else:
            # check that mask length is sufficient
            assert mask.size * utils.mask_bitsize >= len(self)

        self._update_null_count(null_count)

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

    def _update_null_count(self, null_count=None):
        assert null_count is None or null_count >= 0
        if null_count is None:
            if self._mask is not None:
                nnz = libcudf.cudf.count_nonzero_mask(
                    self._mask.mem, size=len(self)
                )
                null_count = len(self) - nnz
                if null_count == 0:
                    self._mask = None
            else:
                null_count = 0

        assert 0 <= null_count <= len(self)
        if null_count == 0:
            # Remove mask if null_count is zero
            self._mask = None

        self._null_count = null_count

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def serialize(self):
        header = {"null_count": self._null_count}
        frames = []

        header["type"] = pickle.dumps(type(self))
        header["dtype"] = self._dtype.str
        header["data_buffer"], data_frames = self._data.serialize()

        header["data_frame_count"] = len(data_frames)
        frames.extend(data_frames)

        if self._mask:
            header["mask_buffer"], mask_frames = self._mask.serialize()
            header["mask_frame_count"] = len(mask_frames)
        else:
            header["mask_buffer"] = []
            header["mask_frame_count"] = 0
            mask_frames = {}

        frames.extend(mask_frames)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        data_nframe = header["data_frame_count"]
        mask_nframe = header["mask_frame_count"]

        data_typ = pickle.loads(header["data_buffer"]["type"])
        data = data_typ.deserialize(
            header["data_buffer"], frames[:data_nframe]
        )

        if header["mask_buffer"]:
            mask_typ = pickle.loads(header["mask_buffer"]["type"])
            mask = mask_typ.deserialize(
                header["mask_buffer"],
                frames[data_nframe : data_nframe + mask_nframe],
            )
        else:
            mask = None
        return data, mask

    def _get_mask_as_column(self):
        from cudf.core.column import NumericalColumn

        data = Buffer(cudautils.ones(len(self), dtype=np.bool_))
        mask = NumericalColumn(
            data=data, mask=None, null_count=0, dtype=np.bool_
        )
        if self._mask is not None:
            mask = mask.set_mask(self._mask).fillna(False)
        return mask

    def __sizeof__(self):
        n = self._data.__sizeof__()
        if self._mask:
            n += self._mask.__sizeof__()
        return n

    def _memory_usage(self, **kwargs):
        return self.__sizeof__()

    def __len__(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data(self):
        """Data buffer
        """
        return self._data

    @property
    def mask(self):
        """Validity mask buffer
        """
        return self._mask

    def set_mask(self, mask, null_count=None):
        """Create new Column by setting the mask

        This will override the existing mask.  The returned Column will
        reference the same data buffer as this Column.

        Parameters
        ----------
        mask : 1D array-like of numpy.uint8
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.

        """
        if not isinstance(mask, Buffer):
            mask = Buffer(mask)
        if mask.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            msg = "mask must be of byte; but got {}".format(mask.dtype)
            raise ValueError(msg)
        return self.replace(mask=mask, null_count=null_count)

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
        return self.set_mask(mask=mask, null_count=0 if all_valid else nelem)

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
        return self.to_dense_buffer(fillna=fillna).to_gpu_array()

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
        return self.to_dense_buffer(fillna=fillna).to_array()

    @property
    def valid_count(self):
        """Number of non-null values"""
        return len(self) - self._null_count

    @property
    def null_count(self):
        """Number of null values"""
        return self._null_count

    @property
    def has_null_mask(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._mask is not None

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        if self.has_null_mask:
            return self._mask
        else:
            raise ValueError("Column has no null mask")

    def _replace_defaults(self):
        params = {
            "data": self.data,
            "mask": self.mask,
            "name": self.name,
            "null_count": self.null_count,
        }
        return params

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
            params = self._replace_defaults()
            return type(self)(**params)

    def replace(self, **kwargs):
        """Replace attributes of the class and return a new Column.

        Valid keywords are valid parameters for ``self.__init__``.
        Any omitted keywords will be defaulted to the corresponding
        attributes in ``self``.
        """
        params = self._replace_defaults()
        params.update(kwargs)
        if "mask" in kwargs and "null_count" not in kwargs:
            del params["null_count"]
        return type(self)(**params)

    def view(self, newcls, **kwargs):
        """View the underlying column data differently using a subclass of
        *TypedColumnBase*.

        Parameters
        ----------
        newcls : TypedColumnBase
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
        val = self.data[index]  # this can raise IndexError
        if isinstance(val, nvstrings.nvstrings):
            val = val.to_host()[0]
        valid = (
            cudautils.mask_get.py_func(self.nullmask, index)
            if self.has_null_mask
            else True
        )
        return val if valid else None

    def __getitem__(self, arg):
        from cudf.core.column import column

        if isinstance(arg, Number):
            arg = int(arg)
            return self.element_indexing(arg)
        elif isinstance(arg, slice):
            # compute mask slice
            if self.null_count > 0:
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError(arg)

                # slicing data
                subdata = self.data[arg]
                # slicing mask
                if self.dtype == "object":
                    data_size = self.data.size()
                else:
                    data_size = self.data.size
                bytemask = cudautils.expand_mask_bits(data_size, self.mask.mem)
                submask = Buffer(cudautils.compact_mask_bytes(bytemask[arg]))
                col = self.replace(data=subdata, mask=submask)
                return col
            else:
                newbuffer = self.data[arg]
                return self.replace(data=newbuffer)
        else:
            arg = column.as_column(arg)
            if len(arg) == 0:
                arg = column.as_column([], dtype="int32")
            if pd.api.types.is_integer_dtype(arg.dtype):
                return self.take(arg.data.mem)
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
                from cudf.core.column import CategoricalColumn
                from cudf.core.buffer import Buffer
                from cudf.utils.cudautils import fill_value

                data = rmm.device_array(nelem, dtype="int8")
                fill_value(data, self._encode(value))
                value = CategoricalColumn(
                    data=Buffer(data),
                    categories=self._categories,
                    ordered=False,
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

        self._data = out.data
        self._mask = out.mask
        self._update_null_count()

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.has_null_mask:
            return self
        out = cudautils.fillna(
            data=self.data.mem, mask=self.mask.mem, value=value
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

    def to_dense_buffer(self, fillna=None):
        """Get dense (no null values) ``Buffer`` of the data.

        Parameters
        ----------
        fillna : scalar, 'pandas', or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        if isinstance(fillna, Number):
            if self.null_count > 0:
                return self.fillna(fillna)
        elif fillna not in {None, "pandas"}:
            raise ValueError("invalid for fillna")

        if self.null_count > 0:
            if fillna == "pandas":
                na_value = self.default_na_value()
                # fill nan
                return self.fillna(na_value)
            else:
                return self._copy_to_dense_buffer()
        else:
            # return a reference for performance reasons, should refactor code
            # to explicitly use mem in the future
            return self.data

    def _invert(self):
        """Internal convenience function for inverting masked array

        Returns
        -------
        DeviceNDArray
           logical inverted mask
        """
        if self.null_count != 0:
            raise ValueError("Column must have no nulls.")
        gpu_mask = self.data.mem
        cudautils.invert_mask(gpu_mask, gpu_mask)
        return self.replace(data=Buffer(gpu_mask), mask=None, null_count=0)

    def _copy_to_dense_buffer(self):
        data = self.data.mem
        mask = self.mask.mem
        nnz, mem = cudautils.copy_to_dense(data=data, mask=mask)
        return Buffer(mem, size=nnz, capacity=mem.size)

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

        return Column._concat([self, as_column(other)])

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

        result.name = self.name
        return result

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """

        if self.null_count != 0:
            raise ValueError("Column must have no nulls.")

        return cudautils.compact_mask_bytes(self.data.mem)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @property
    def _pointer(self):
        """
        Return pointer to a view of the underlying data structure
        """
        return libcudf.cudf.column_view_pointer(self)

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
            return self.find_first_value(label)
        if side == "right":
            return self.find_last_value(label) + 1

    def sort_by_values(self):
        raise NotImplementedError

    def _unique_segments(self):
        """ Common code for unique, unique_count and value_counts"""
        # make dense column
        densecol = self.replace(data=self.to_dense_buffer(), mask=None)
        # sort the column
        sortcol, _ = densecol.sort_by_values()
        # find segments
        sortedvals = sortcol.data.mem
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


class TypedColumnBase(Column):
    """Base class for all typed column
    e.g. NumericalColumn, CategoricalColumn

    This class provides common operations to implement logical view and
    type-based operations for the column.

    Notes
    -----
    Not designed to be instantiated directly.  Instantiate subclasses instead.
    """

    def __init__(self, **kwargs):
        dtype = kwargs.pop("dtype")
        super(TypedColumnBase, self).__init__(**kwargs)
        # Logical dtype
        self._dtype = pd.api.types.pandas_dtype(dtype)

    @property
    def dtype(self):
        return self._dtype

    def is_type_equivalent(self, other):
        """Is the logical type of the column equal to the other column.
        """
        mine = self._replace_defaults()
        theirs = other._replace_defaults()

        def remove_base(dct):
            # removes base attributes in the phyiscal layer.
            basekeys = Column._replace_defaults(self).keys()
            for k in basekeys:
                del dct[k]

        remove_base(mine)
        remove_base(theirs)

        # Check categories via Column.equals(). Pop them off the
        # dicts so the == below doesn't try to invoke `__eq__()`
        if ("categories" in mine) or ("categories" in theirs):
            if "categories" not in mine:
                return False
            if "categories" not in theirs:
                return False
            if not mine.pop("categories").equals(theirs.pop("categories")):
                return False

        return type(self) == type(other) and mine == theirs

    def _replace_defaults(self):
        params = super(TypedColumnBase, self)._replace_defaults()
        params.update(dict(dtype=self._dtype))
        return params

    def _mimic_inplace(self, result, inplace=False):
        """
        If `inplace=True`, used to mimic an inplace operation
        by replacing data in ``self`` with data in ``result``.

        Otherwise, returns ``result`` unchanged.
        """
        if inplace:
            self._data = result._data
            self._mask = result._mask
            self._null_count = result._null_count
        else:
            return result

    def argsort(self, ascending):
        _, inds = self.sort_by_values(ascending=ascending)
        return inds

    def sort_by_values(self, ascending):
        raise NotImplementedError

    def find_and_replace(self, to_replace, values):
        raise NotImplementedError

    def dropna(self):
        dropped_col = libcudf.stream_compaction.drop_nulls([self])
        if not dropped_col:
            return column_empty_like(self, newsize=0)
        else:
            return self.replace(
                data=dropped_col[0].data, mask=None, null_count=0
            )

    def apply_boolean_mask(self, mask):

        mask = as_column(mask, dtype="bool")
        data = libcudf.stream_compaction.apply_boolean_mask([self], mask)
        if not data:
            return column_empty_like(self, newsize=0)
        else:
            return self.replace(
                data=data[0].data,
                mask=data[0].mask,
                null_count=data[0].null_count,
            )

    def fillna(self, fill_value, inplace):
        raise NotImplementedError

    def searchsorted(self, value, side="left"):
        raise NotImplementedError

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
        ) and self.null_count > 0:
            cats = cats.dropna()
            labels = labels - 1

        return cudf.core.column.CategoricalColumn(
            data=labels._column.data,
            mask=self.mask,
            null_count=self.null_count,
            categories=cats._column,
            ordered=ordered,
        )
        raise NotImplementedError

    def as_numerical_column(self, dtype, **kwargs):
        raise NotImplementedError

    def as_datetime_column(self, dtype, **kwargs):
        raise NotImplementedError

    def as_string_column(self, dtype, **kwargs):
        raise NotImplementedError

    @property
    def __cuda_array_interface__(self):
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data.mem.device_ctypes_pointer.value, True),
            "version": 1,
        }

        if self.has_null_mask:
            from types import SimpleNamespace

            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = SimpleNamespace(
                __cuda_array_interface__={
                    "shape": (len(self),),
                    "typestr": "<t1",
                    "data": (
                        self.nullmask.mem.device_ctypes_pointer.value,
                        True,
                    ),
                    "version": 1,
                }
            )
            output["mask"] = mask

        return output


def column_empty_like(column, dtype=None, masked=False, newsize=None):
    """Allocate a new column like the given *column*
    """
    if dtype is None:
        dtype = column.dtype
    row_count = len(column) if newsize is None else newsize
    categories = None
    if is_categorical_dtype(dtype):
        categories = column.cat().categories
        dtype = column.data.dtype
    return column_empty(row_count, dtype, masked, categories=categories)


def column_empty_like_same_mask(column, dtype):
    """Create a new empty Column with the same length and the same mask.

    Parameters
    ----------
    dtype : np.dtype like
        The dtype of the data buffer.
    """
    result = column_empty_like(column, dtype)
    if column.has_null_mask:
        result = result.set_mask(column.mask)
    return result


def column_empty(row_count, dtype, masked, categories=None):
    """Allocate a new column like the given row_count and dtype.
    """
    dtype = pd.api.types.pandas_dtype(dtype)
    if masked:
        mask = cudautils.make_empty_mask(row_count)
    else:
        mask = None

    if categories is None and is_categorical_dtype(dtype):
        categories = [] if dtype.categories is None else dtype.categories

    if categories is not None:
        mem = rmm.device_array((row_count,), dtype=dtype)
        data = Buffer(mem)
        dtype = "category"
    elif dtype.kind in "OU":
        if row_count == 0:
            data = nvstrings.to_device([])
        else:
            mem = rmm.device_array((row_count,), dtype="float64")
            data = nvstrings.dtos(mem, len(mem), nulls=mask, bdevmem=True)
    else:
        mem = rmm.device_array((row_count,), dtype=dtype)
        data = Buffer(mem)

    if mask is not None:
        mask = Buffer(mask)

    from cudf.core.column import build_column

    return build_column(data, dtype, mask, categories)


def build_column(
    buffer, dtype, mask=None, categories=None, name=None, null_count=None
):
    from cudf.core.column import numerical, categorical, datetime, string

    dtype = pd.api.types.pandas_dtype(dtype)
    if is_categorical_dtype(dtype):
        return categorical.CategoricalColumn(
            data=buffer,
            dtype="categorical",
            categories=categories,
            ordered=False,
            mask=mask,
            name=name,
            null_count=null_count,
        )
    elif dtype.type is np.datetime64:
        return datetime.DatetimeColumn(
            data=buffer,
            dtype=dtype,
            mask=mask,
            name=name,
            null_count=null_count,
        )
    elif dtype.type in (np.object_, np.str_):
        if not isinstance(buffer, nvstrings.nvstrings):
            raise TypeError
        return string.StringColumn(
            data=buffer, name=name, null_count=null_count
        )
    else:
        return numerical.NumericalColumn(
            data=buffer,
            dtype=dtype,
            mask=mask,
            name=name,
            null_count=null_count,
        )


def as_column(arbitrary, nan_as_null=True, dtype=None, name=None):
    """Create a Column from an arbitrary object
    Currently support inputs are:
    * ``Column``
    * ``Buffer``
    * ``Series``
    * ``Index``
    * numba device array
    * cuda array interface
    * numpy array
    * pyarrow array
    * pandas.Categorical
    * Object exposing ``__cuda_array_interface__``
    Returns
    -------
    result : subclass of TypedColumnBase
        - CategoricalColumn for pandas.Categorical input.
        - DatetimeColumn for datetime input.
        - StringColumn for string input.
        - NumericalColumn for all other inputs.
    """
    from cudf.core.column import numerical, categorical, datetime, string
    from cudf.core.series import Series
    from cudf.core.index import Index

    if name is None and hasattr(arbitrary, "name"):
        name = arbitrary.name

    if isinstance(arbitrary, Column):
        return arbitrary

    elif isinstance(arbitrary, Series):
        data = arbitrary._column
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, Index):
        data = arbitrary._values
        if dtype is not None:
            data = data.astype(dtype)
    elif isinstance(arbitrary, Buffer):
        data = numerical.NumericalColumn(data=arbitrary, dtype=arbitrary.dtype)

    elif isinstance(arbitrary, nvstrings.nvstrings):
        data = string.StringColumn(data=arbitrary)

    elif cuda.devicearray.is_cuda_ndarray(arbitrary):
        data = as_column(Buffer(arbitrary))
        if (
            data.dtype in [np.float16, np.float32, np.float64]
            and arbitrary.size > 0
        ):
            if nan_as_null:
                mask = libcudf.unaryops.nans_to_nulls(data)
                data = data.set_mask(mask)

        elif data.dtype.kind == "M":
            null = cudf.core.column.column_empty_like(
                data, masked=True, newsize=1
            )
            col = libcudf.replace.replace(
                as_column(Buffer(arbitrary)),
                as_column(
                    Buffer(np.array([np.datetime64("NaT")], dtype=data.dtype))
                ),
                null,
            )
            data = datetime.DatetimeColumn(
                data=Buffer(arbitrary), mask=col.mask, dtype=data.dtype
            )

    elif hasattr(arbitrary, "__cuda_array_interface__"):
        desc = arbitrary.__cuda_array_interface__
        data = _data_from_cuda_array_interface_desc(desc)
        mask = _mask_from_cuda_array_interface_desc(desc)

        if mask is not None:
            nelem = len(data.mem)
            nnz = libcudf.cudf.count_nonzero_mask(mask.mem, size=nelem)
            null_count = nelem - nnz
        else:
            null_count = 0
        col = build_column(
            data, dtype=data.dtype, mask=mask, name=name, null_count=null_count
        )
        # Keep a reference to `arbitrary` with the underlying
        # RMM device array, so that the memory isn't freed out
        # from under us
        col.data.mem._obj = arbitrary
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
            count = len(arbitrary)
            null_count = arbitrary.null_count

            buffers = arbitrary.buffers()
            # Buffer of actual strings values
            if buffers[2] is not None:
                sbuf = np.frombuffer(buffers[2], dtype="int8")
            else:
                sbuf = np.empty(0, dtype="int8")
            # Buffer of offsets values
            obuf = np.frombuffer(buffers[1], dtype="int32")
            # Buffer of null bitmask
            nbuf = None
            if null_count > 0:
                nbuf = np.frombuffer(buffers[0], dtype="int8")

            data = as_column(
                nvstrings.from_offsets(
                    sbuf, obuf, count, nbuf=nbuf, ncount=null_count
                )
            )
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
            pamask, padata = buffers_from_pyarrow(arbitrary)
            data = categorical.CategoricalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                categories=arbitrary.dictionary,
                ordered=arbitrary.type.ordered,
            )
        elif isinstance(arbitrary, pa.TimestampArray):
            dtype = np.dtype("M8[{}]".format(arbitrary.type.unit))
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype=dtype)
            data = datetime.DatetimeColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=dtype,
            )
        elif isinstance(arbitrary, pa.Date64Array):
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype="M8[ms]")
            data = datetime.DatetimeColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype("M8[ms]"),
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
            pamask, padata = buffers_from_pyarrow(arbitrary, dtype=dtype)
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=dtype,
            )
        else:
            pamask, padata = buffers_from_pyarrow(arbitrary)
            data = numerical.NumericalColumn(
                data=padata,
                mask=pamask,
                null_count=arbitrary.null_count,
                dtype=np.dtype(arbitrary.type.to_pandas_dtype()),
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

        data = Column._concat(gpu_cols, dtype=new_dtype)

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
        if hasattr(arbitrary, "dtype"):
            data_type = np_to_pa_dtype(arbitrary.dtype)
            # PyArrow can't construct date64 or date32 arrays from np
            # datetime types
            if pa.types.is_date64(data_type) or pa.types.is_date32(data_type):
                arbitrary = arbitrary.astype("int64")
            data = as_column(pa.array([arbitrary], type=data_type))
        else:
            data = as_column(pa.array([arbitrary]), nan_as_null=nan_as_null)

    elif isinstance(arbitrary, memoryview):
        data = as_column(
            np.array(arbitrary), dtype=dtype, nan_as_null=nan_as_null
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
                        np.array(arbitrary, dtype=np_type),
                        nan_as_null=nan_as_null,
                    )
    if hasattr(data, "name") and (name is not None):
        data.name = name
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
    result : Buffer
    """
    core = njit(udf)
    results = rmm.device_array(shape=len(column), dtype=out_dtype)
    values = column.data.mem
    if column.mask:
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

        masks = column.mask.mem
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
    # Output
    return Buffer(results)


def _data_from_cuda_array_interface_desc(desc):
    ptr = desc["data"][0]
    nelem = desc["shape"][0]
    dtype = np.dtype(desc["typestr"])

    data = rmm.device_array_from_ptr(
        ptr, nelem=nelem, dtype=dtype, finalizer=None
    )
    data = Buffer(data)
    return data


def _mask_from_cuda_array_interface_desc(desc):
    from cudf.utils.utils import calc_chunk_size, mask_dtype, mask_bitsize
    from cudf.utils.cudautils import compact_mask_bytes

    mask = desc.get("mask", None)

    if mask is not None:
        desc = mask.__cuda_array_interface__
        ptr = desc["data"][0]
        nelem = desc["shape"][0]
        typestr = desc["typestr"]
        typecode = typestr[1]
        if typecode == "t":
            mask = rmm.device_array_from_ptr(
                ptr,
                nelem=calc_chunk_size(nelem, mask_bitsize),
                dtype=mask_dtype,
                finalizer=None,
            )
            mask = Buffer(mask)
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
