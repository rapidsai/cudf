# Copyright (c) 2018, NVIDIA CORPORATION.

"""
A column is data + validity-mask.
LibGDF operates on column.
"""
import pickle
from numbers import Number

import numpy as np
import pandas as pd

import nvstrings
from librmm_cffi import librmm as rmm

import cudf.bindings.quantile as cpp_quantile
from cudf.bindings.concat import _column_concat
from cudf.bindings.cudf_cpp import column_view_pointer, count_nonzero_mask
from cudf.dataframe.buffer import Buffer
from cudf.utils import cudautils, ioutils, utils
from cudf.utils.dtypes import is_categorical_dtype


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
        from cudf.dataframe.series import Series
        from cudf.dataframe.string import StringColumn
        from cudf.dataframe.categorical import CategoricalColumn
        from cudf.dataframe.numerical import NumericalColumn

        if len(objs) == 0:
            dtype = pd.api.types.pandas_dtype(dtype)
            if dtype.type in (np.object_, np.str_):
                return StringColumn(data=nvstrings.to_device([]), null_count=0)
            elif is_categorical_dtype(dtype):
                return CategoricalColumn(
                    data=Column(Buffer.null(np.dtype("int8"))),
                    null_count=0,
                    ordered=False,
                )
            else:
                return Column(Buffer.null(dtype))

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
                    from cudf.dataframe.columnops import column_empty_like

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
            return StringColumn(data=nvstrings.from_strings(*objs))

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
            col = _column_concat(objs, col)

        return col

    @staticmethod
    def from_mem_views(data_mem, mask_mem=None, null_count=None, name=None):
        """Create a Column object from a data device array (or nvstrings
           object), and an optional mask device array
        """
        from cudf.dataframe import columnops

        if isinstance(data_mem, nvstrings.nvstrings):
            return columnops.build_column(
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
            return columnops.build_column(
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
                nnz = count_nonzero_mask(self._mask.mem, size=len(self))
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
        from cudf.dataframe.numerical import NumericalColumn

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
            import cudf.bindings.copying as cpp_copy

            return cpp_copy.copy_column(self)
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
        from cudf.dataframe import columnops

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
                bytemask = cudautils.expand_mask_bits(
                    data_size, self.mask.to_gpu_array()
                )
                submask = Buffer(cudautils.compact_mask_bytes(bytemask[arg]))
                col = self.replace(data=subdata, mask=submask)
                return col
            else:
                newbuffer = self.data[arg]
                return self.replace(data=newbuffer)
        else:
            arg = columnops.as_column(arg)
            if len(arg) == 0:
                arg = columnops.as_column([], dtype="int32")
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
        import cudf.bindings.copying as cpp_copying
        from cudf.dataframe import columnops

        if isinstance(key, slice):
            key_start, key_stop, key_stride = key.indices(len(self))
            if key_stride != 1:
                raise NotImplementedError("Stride not supported in slice")
            nelem = abs(key_stop - key_start)
        else:
            key = columnops.as_column(key)
            if pd.api.types.is_bool_dtype(key.dtype):
                if not len(key) == len(self):
                    raise ValueError(
                        "Boolean mask must be of same length as column"
                    )
                key = columnops.as_column(cudautils.arange(len(self)))[key]
            nelem = len(key)

        if utils.is_scalar(value):
            if is_categorical_dtype(self.dtype):
                from cudf.dataframe.categorical import CategoricalColumn
                from cudf.dataframe.buffer import Buffer
                from cudf.utils.cudautils import fill_value

                data = rmm.device_array(nelem, dtype="int8")
                fill_value(data, self._encode(value))
                value = CategoricalColumn(
                    data=Buffer(data),
                    categories=self._categories,
                    ordered=False,
                )
            elif value is None:
                value = columnops.column_empty(nelem, self.dtype, masked=True)
            else:
                to_dtype = pd.api.types.pandas_dtype(self.dtype)
                value = utils.scalar_broadcast_to(value, nelem, to_dtype)

        value = columnops.as_column(value).astype(self.dtype)

        if len(value) != nelem:
            msg = (
                f"Size mismatch: cannot set value "
                f"of size {len(value)} to indexing result of size "
                f"{nelem}"
            )
            raise ValueError(msg)

        if isinstance(key, slice):
            out = cpp_copying.apply_copy_range(
                self, value, key_start, key_stop, 0
            )
        else:
            out = cpp_copying.apply_scatter(value, key, self)

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
            data=self.data.to_gpu_array(),
            mask=self.mask.to_gpu_array(),
            value=value,
        )
        return self.replace(data=Buffer(out), mask=None, null_count=0)

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

        gpu_mask = self.to_gpu_array()
        cudautils.invert_mask(gpu_mask, gpu_mask)
        return self.replace(data=Buffer(gpu_mask), mask=None, null_count=0)

    def _copy_to_dense_buffer(self):
        data = self.data.to_gpu_array()
        mask = self.mask.to_gpu_array()
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
        from cudf.dataframe.columnops import as_column

        return Column._concat([self, as_column(other)])

    def quantile(self, q, interpolation, exact):
        if isinstance(q, Number):
            quant = [float(q)]
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            quant = q
        else:
            msg = "`q` must be either a single element, list or numpy array"
            raise TypeError(msg)
        return cpp_quantile.apply_quantile(self, quant, interpolation, exact)

    def take(self, indices, ignore_index=False):
        """Return Column by taking values from the corresponding *indices*.
        """
        import cudf.bindings.copying as cpp_copying
        from cudf.dataframe.columnops import column_empty_like

        indices = Buffer(indices).to_gpu_array()
        # Handle zero size
        if indices.size == 0:
            return column_empty_like(self, newsize=0)

        # Returns a new column
        result = cpp_copying.apply_gather(self, indices)
        result.name = self.name
        return result

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """
        return cudautils.compact_mask_bytes(self.to_gpu_array())

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
        return column_view_pointer(self)

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
        segs, _ = self._unique_segments()
        if dropna is False and self.null_count > 0:
            return len(segs) + 1
        return len(segs)
