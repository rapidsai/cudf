# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.limits cimport INT_MAX
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column, column_contents
from pylibcudf.libcudf.column.column_factories cimport make_column_from_scalar
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from .gpumemoryview cimport gpumemoryview
from .filling cimport sequence
from .scalar cimport Scalar
from .types cimport DataType, size_of, type_id
from .utils cimport int_to_bitmask_ptr, int_to_void_ptr

from functools import cache


try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except ImportError:
    cp = None


__all__ = ["Column", "ListColumnView", "is_c_contiguous"]

cdef class Column:
    """A container of nullable device data as a column of elements.

    This class is an implementation of `Arrow columnar data specification
    <https://arrow.apache.org/docs/format/Columnar.html>`__ for data stored on
    GPUs. It relies on Python memoryview-like semantics to maintain shared
    ownership of the data it is constructed with, so any input data may also be
    co-owned by other data structures. The Column is designed to be operated on
    using algorithms backed by libcudf.

    Parameters
    ----------
    data_type : DataType
        The type of data in the column.
    size : size_type
        The number of rows in the column.
    data : gpumemoryview
        The data the column will refer to.
    mask : gpumemoryview
        The null mask for the column.
    null_count : int
        The number of null rows in the column.
    offset : int
        The offset into the data buffer where the column's data begins.
    children : list
        The children of this column if it is a compound column type.
    """
    def __init__(
        self, DataType data_type not None, size_type size, gpumemoryview data,
        gpumemoryview mask, size_type null_count, size_type offset,
        list children
    ):
        if not all(isinstance(c, Column) for c in children):
            raise ValueError("All children must be pylibcudf Column objects")
        self._data_type = data_type
        self._size = size
        self._data = data
        self._mask = mask
        self._null_count = null_count
        self._offset = offset
        self._children = children
        self._num_children = len(children)

    __hash__ = None

    cdef column_view view(self) nogil:
        """Generate a libcudf column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL

        if self._data is not None:
            data = int_to_void_ptr(self._data.ptr)
        if self._mask is not None:
            null_mask = int_to_bitmask_ptr(self._mask.ptr)

        # TODO: Check if children can ever change. If not, this could be
        # computed once in the constructor and always be reused.
        cdef vector[column_view] c_children
        with gil:
            if self._children is not None:
                for child in self._children:
                    # Need to cast to Column here so that Cython knows that
                    # `view` returns a typed object, not a Python object. We
                    # cannot use a typed variable for `child` because cdef
                    # declarations cannot be inside nested blocks (`if` or
                    # `with` blocks) so we cannot declare it inside the `with
                    # gil` block, but we also cannot declare it outside the
                    # `with gil` block because it is erroneous to declare a
                    # variable of a cdef class type in a `nogil` context (which
                    # this whole function is).
                    c_children.push_back((<Column> child).view())

        return column_view(
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
        )

    cdef mutable_column_view mutable_view(self) nogil:
        """Generate a libcudf mutable_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef void * data = NULL
        cdef bitmask_type * null_mask = NULL

        if self._data is not None:
            data = int_to_void_ptr(self._data.ptr)
        if self._mask is not None:
            null_mask = int_to_bitmask_ptr(self._mask.ptr)

        cdef vector[mutable_column_view] c_children
        with gil:
            if self._children is not None:
                for child in self._children:
                    # See the view method for why this needs to be cast.
                    c_children.push_back((<Column> child).mutable_view())

        return mutable_column_view(
            self._data_type.c_obj, self._size, data, null_mask,
            self._null_count, self._offset, c_children
        )

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col):
        """Create a Column from a libcudf column.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef DataType dtype = DataType.from_libcudf(libcudf_col.get().type())
        cdef size_type size = libcudf_col.get().size()

        cdef size_type null_count = libcudf_col.get().null_count()

        cdef column_contents contents = libcudf_col.get().release()

        # Note that when converting to cudf Column objects we'll need to pull
        # out the base object.
        cdef gpumemoryview data = gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(contents.data))
        )

        cdef gpumemoryview mask = None
        if null_count > 0:
            mask = gpumemoryview(
                DeviceBuffer.c_from_unique_ptr(move(contents.null_mask))
            )

        children = []
        if contents.children.size() != 0:
            for i in range(contents.children.size()):
                children.append(
                    Column.from_libcudf(move(contents.children[i]))
                )

        return Column(
            dtype,
            size,
            data,
            mask,
            null_count,
            # Initial offset when capturing a C++ column is always 0.
            0,
            children,
        )

    cpdef Column with_mask(self, gpumemoryview mask, size_type null_count):
        """Augment this column with a new null mask.

        Parameters
        ----------
        mask : gpumemoryview
            New mask (or None to unset the mask)
        null_count : int
            New null count. If this is incorrect, bad things happen.

        Returns
        -------
        New Column object sharing data with self (except for the mask which is new).
        """
        if mask is None and null_count > 0:
            raise ValueError("Empty mask must have null count of zero")
        return Column(
            self._data_type,
            self._size,
            self._data,
            mask,
            null_count,
            self._offset,
            self._children,
        )

    @staticmethod
    cdef Column from_column_view(const column_view& cv, Column owner):
        """Create a Column from a libcudf column_view into a Column owner.

        This method accepts shared ownership of the underlying data from the
        owner and relies on the offset from the view.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        children = []
        cdef size_type i
        if cv.num_children() != 0:
            for i in range(cv.num_children()):
                children.append(Column.from_column_view(cv.child(i), owner.child(i)))

        return Column(
            DataType.from_libcudf(cv.type()),
            cv.size(),
            owner._data,
            owner._mask,
            cv.null_count(),
            cv.offset(),
            children,
        )

    # Ideally this function would simply be handled via a fused type in
    # from_column_view, but this does not work due to
    # https://github.com/cython/cython/issues/6740
    @staticmethod
    cdef Column from_column_view_of_arbitrary(const column_view& cv, object owner):
        """Create a Column from a libcudf column_view into an arbitrary owner.

        This method accepts shared ownership of the underlying data from the owner.
        Since the owner may be any arbitrary object, every child Column also shares
        ownership of the same buffer since they do not have the information available to
        choose to only own subsets of it.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # For efficiency, prohibit calling this overload with a Column owner.
        assert not isinstance(owner, Column)

        children = []
        cdef size_type i
        if cv.num_children() != 0:
            for i in range(cv.num_children()):
                children.append(
                    Column.from_column_view_of_arbitrary(cv.child(i), owner)
                )

        cdef gpumemoryview owning_data = gpumemoryview.from_pointer(
            <Py_ssize_t> cv.head[char](), owner
        )
        cdef gpumemoryview owning_mask = gpumemoryview.from_pointer(
            <Py_ssize_t> cv.null_mask(), owner
        )

        return Column(
            DataType.from_libcudf(cv.type()),
            cv.size(),
            owning_data,
            owning_mask,
            cv.null_count(),
            cv.offset(),
            children,
        )

    @staticmethod
    def from_scalar(Scalar slr, size_type size):
        """Create a Column from a Scalar.

        Parameters
        ----------
        slr : Scalar
            The scalar to create a column from.
        size : size_type
            The number of elements in the column.

        Returns
        -------
        Column
            A Column containing the scalar repeated `size` times.
        """
        cdef const scalar* c_scalar = slr.get()
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = make_column_from_scalar(dereference(c_scalar), size)
        return Column.from_libcudf(move(c_result))

    @staticmethod
    def all_null_like(Column like, size_type size):
        """Create an all null column from a template.

        Parameters
        ----------
        like : Column
            Column whose type we should mimic
        size : int
            Number of rows in the resulting column.

        Returns
        -------
        Column
            An all-null column of `size` rows and type matching `like`.
        """
        cdef Scalar slr = Scalar.empty_like(like)
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = make_column_from_scalar(dereference(slr.get()), size)
        return Column.from_libcudf(move(c_result))

    @classmethod
    def from_array_interface(cls, obj):
        """
        Create a Column from an object implementing the NumPy Array Interface.

        Parameters
        ----------
        obj : object
            Must implement the `__array_interface__` protocol.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError(
            "Converting to a pylibcudf Column is not yet implemented."
        )

    @classmethod
    def from_cuda_array_interface(cls, obj):
        """
        Create a Column from an object implementing the CUDA Array Interface.

        Parameters
        ----------
        obj : object
            Must implement the `__cuda_array_interface__` protocol.

        Returns
        -------
        Column
            A 1D or 2D list column, depending on the input shape.

        Raises
        ------
        ValueError
            If the object is not 1D or 2D, or is not C-contiguous.
        ImportError
            If CuPy is not installed.
        """
        if np is None or cp is None:
            raise ImportError("NumPy and CuPy must be installed to use this method")

        iface = getattr(obj, "__cuda_array_interface__", None)
        if iface is None:
            raise TypeError("Object does not implement __cuda_array_interface__")

        if iface.get("mask") is not None:
            raise ValueError("mask not yet supported")

        typestr = iface["typestr"][1:]
        data_type = _datatype_from_dtype_desc(typestr)

        shape = iface["shape"]
        if not is_c_contiguous(shape, iface["strides"], size_of(data_type)):
            raise ValueError("Data must be C-contiguous")

        if len(shape) == 1:
            data = gpumemoryview(obj)
            size = shape[0]
            return cls(data_type, size, data, None, 0, 0, [])
        elif len(shape) == 2:
            arr = obj
            flat_data = cp.ravel(arr)
            num_rows, num_cols = shape
            if num_rows < INT_MAX:
                offsets_col = sequence(
                    num_rows + 1,
                    Scalar.from_numpy(np.int32(0)),
                    Scalar.from_numpy(np.int32(num_cols))
                )
            else:
                raise ValueError(
                    "Number of rows exceeds int32 limit for offsets column."
                )
            data_col = cls(
                data_type=data_type,
                size=flat_data.size,
                data=gpumemoryview(flat_data),
                mask=None,
                null_count=0,
                offset=0,
                children=[],
            )
            return cls(
                data_type=DataType(type_id.LIST),
                size=num_rows,
                data=None,
                mask=None,
                null_count=0,
                offset=0,
                children=[offsets_col, data_col],
            )
        else:
            raise ValueError("Only 1D or 2D arrays are supported")

    @classmethod
    def from_arraylike(cls, obj):
        """
        Create a Column from any object which supports the NumPy
        or CUDA array interface.

        Parameters
        ----------
        obj : object
            The input array to be converted into a `pylibcudf.Column`.

        Returns
        -------
        Column

        Raises
        ------
        TypeError
            If the input does not implement a supported array interface.
        ImportError
            If NumPy or CuPy is required but not installed.

        Notes
        -----
        - For `cupy.ndarray`, this uses the CUDA array interface and
          supports 1D or 2D arrays.
        - For `numpy.ndarray`, this is not yet implemented.

        Examples:
            >>> import pylibcudf as plc
            >>> import cupy as cp
            >>> cp_arr = cp.array([[1,2],[3,4]])
            >>> col = plc.Column.from_arraylike(cp_arr)
        """
        if hasattr(obj, "__cuda_array_interface__"):
            return cls.from_cuda_array_interface(obj)
        if hasattr(obj, "__array_interface__"):
            return cls.from_array_interface(obj)

        raise TypeError(
            f"Cannot convert object of type {type(obj)} to a pylibcudf Column"
        )

    cpdef DataType type(self):
        """The type of data in the column."""
        return self._data_type

    cpdef Column child(self, size_type index):
        """Get a child column of this column.

        Parameters
        ----------
        index : size_type
            The index of the child column to get.

        Returns
        -------
        Column
            The child column.
        """
        return self._children[index]

    cpdef size_type num_children(self):
        """The number of children of this column."""
        return self._num_children

    cpdef ListColumnView list_view(self):
        """Accessor for methods of a Column that are specific to lists."""
        return ListColumnView(self)

    cpdef gpumemoryview data(self):
        """The data buffer of the column."""
        return self._data

    cpdef gpumemoryview null_mask(self):
        """The null mask of the column."""
        return self._mask

    cpdef size_type size(self):
        """The number of elements in the column."""
        return self._size

    cpdef size_type offset(self):
        """The offset of the column."""
        return self._offset

    cpdef size_type null_count(self):
        """The number of null elements in the column."""
        return self._null_count

    cpdef list children(self):
        """The children of the column."""
        return self._children

    cpdef Column copy(self):
        """Create a copy of the column."""
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = make_unique[column](self.view())
        return Column.from_libcudf(move(c_result))


cdef class ListColumnView:
    """Accessor for methods of a Column that are specific to lists."""
    def __init__(self, Column col):
        if col.type().id() != type_id.LIST:
            raise TypeError("Column is not a list type")
        self._column = col

    __hash__ = None

    cpdef child(self):
        """The data column of the underlying list column."""
        return self._column.child(1)

    cpdef offsets(self):
        """The offsets column of the underlying list column."""
        return self._column.child(1)

    cdef lists_column_view view(self) nogil:
        """Generate a libcudf lists_column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        return lists_column_view(self._column.view())


@cache
def _datatype_from_dtype_desc(desc):
    mapping = {
        'u1': type_id.UINT8,
        'u2': type_id.UINT16,
        'u4': type_id.UINT32,
        'u8': type_id.UINT64,
        'i1': type_id.INT8,
        'i2': type_id.INT16,
        'i4': type_id.INT32,
        'i8': type_id.INT64,
        'f4': type_id.FLOAT32,
        'f8': type_id.FLOAT64,
        'b1': type_id.BOOL8,
        'M8[s]': type_id.TIMESTAMP_SECONDS,
        'M8[ms]': type_id.TIMESTAMP_MILLISECONDS,
        'M8[us]': type_id.TIMESTAMP_MICROSECONDS,
        'M8[ns]': type_id.TIMESTAMP_NANOSECONDS,
        'm8[s]': type_id.DURATION_SECONDS,
        'm8[ms]': type_id.DURATION_MILLISECONDS,
        'm8[us]': type_id.DURATION_MICROSECONDS,
        'm8[ns]': type_id.DURATION_NANOSECONDS,
    }
    if desc not in mapping:
        raise ValueError(f"Unsupported dtype: {desc}")
    return DataType(mapping[desc])


def is_c_contiguous(
    shape: Sequence[int], strides: None | Sequence[int], itemsize: int
) -> bool:
    """Determine if shape and strides are C-contiguous

    Parameters
    ----------
    shape : Sequence[int]
        Number of elements in each dimension.
    strides : None | Sequence[int]
        The stride of each dimension in bytes.
        If None, the memory layout is C-contiguous.
    itemsize : int
        Size of an element in bytes.

    Returns
    -------
    bool
        The boolean answer.
    """

    if strides is None:
        return True
    if any(dim == 0 for dim in shape):
        return True
    cumulative_stride = itemsize
    for dim, stride in zip(reversed(shape), reversed(strides)):
        if dim > 1 and stride != cumulative_stride:
            return False
        cumulative_stride *= dim
    return True
