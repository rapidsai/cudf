# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from cython.operator cimport dereference

from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_New,
)

from libc.stdint cimport uintptr_t

from libcpp.limits cimport numeric_limits
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from pylibcudf.libcudf.column.column cimport column, column_contents
from pylibcudf.libcudf.column.column_factories cimport make_column_from_scalar
from pylibcudf.libcudf.interop cimport (
    ArrowArray,
    ArrowSchema,
    ArrowDeviceArray,
    arrow_column,
    column_metadata,
    to_arrow_host_raw,
    to_arrow_device_raw,
    to_arrow_schema_raw,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar, numeric_scalar
from pylibcudf.libcudf.types cimport size_type, size_of as cpp_size_of, bitmask_type
from pylibcudf.libcudf.utilities.traits cimport is_fixed_width
from pylibcudf.libcudf.copying cimport get_element


from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream

from .gpumemoryview cimport gpumemoryview
from .filling cimport sequence
from .scalar cimport Scalar
from .traits cimport (
    is_fixed_width as plc_is_fixed_width,
    is_nested,
)
from .types cimport DataType, size_of, type_id
from ._interop_helpers cimport (
    _release_schema,
    _release_array,
    _release_device_array,
    _metadata_to_libcudf,
)
from .null_mask cimport bitmask_allocation_size_bytes
from .utils cimport _get_stream

from .gpumemoryview import _datatype_from_dtype_desc
from ._interop_helpers import ColumnMetadata

import functools

__all__ = ["Column", "ListColumnView", "is_c_contiguous"]


class _ArrowLikeMeta(type):
    def __subclasscheck__(cls, other):
        # We cannot separate these types via singledispatch because the dispatch
        # will often be ambiguous when objects expose multiple protocols.
        return (
            hasattr(other, "__arrow_c_array__")
            or hasattr(other, "__arrow_c_device_array__")
        )


class _ArrowLike(metaclass=_ArrowLikeMeta):
    pass


cdef class _ArrowColumnHolder:
    """A holder for an Arrow column for gpumemoryview lifetime management."""
    cdef unique_ptr[arrow_column] col


cdef class OwnerWithCAI:
    """An interface for column view's data with gpumemoryview via CAI."""
    @staticmethod
    cdef create(column_view cv, object owner):
        obj = OwnerWithCAI()
        obj.owner = owner
        cdef int size
        cdef column_view offsets_column
        cdef unique_ptr[scalar] last_offset
        if cv.type().id() == type_id.EMPTY:
            size = cv.size()
        elif is_fixed_width(cv.type()):
            size = cv.size() * cpp_size_of(cv.type())
        elif cv.type().id() == type_id.STRING:
            # The size of the character array in the parent is the offsets size
            num_children = cv.num_children()
            size = 0
            # A strings column with no children is created for empty/all null
            if num_children:
                offsets_column = cv.child(0)
                last_offset = get_element(offsets_column, offsets_column.size() - 1)
                size = (<numeric_scalar[size_type] *> last_offset.get()).value()
        else:
            # All other types store data in the children, so the parent size is 0
            size = 0

        obj.cai = {
            "shape": (size,),
            "strides": None,
            # For the purposes in this function, just treat all of the types as byte
            # streams of the appropriate size. This matches what we currently get from
            # rmm.DeviceBuffer
            "typestr": "|u1",
            "data": (<uintptr_t> cv.head[char](), False),
            "version": 3,
        }
        return obj

    @property
    def __cuda_array_interface__(self):
        return self.cai


cdef class OwnerMaskWithCAI:
    """An interface for column view's null mask with gpumemoryview via CAI."""
    @staticmethod
    cdef create(column_view cv, object owner):
        obj = OwnerMaskWithCAI()
        obj.owner = owner

        obj.cai = {
            "shape": (bitmask_allocation_size_bytes(cv.size()),),
            "strides": None,
            # For the purposes in this function, just treat all of the types as byte
            # streams of the appropriate size. This matches what we currently get from
            # rmm.DeviceBuffer
            "typestr": "|u1",
            "data": (<uintptr_t> cv.null_mask(), False),
            "version": 3,
        }
        return obj

    @property
    def __cuda_array_interface__(self):
        return self.cai


class _Ravelled:
    def __init__(self, obj):
        self.obj = obj
        cai = obj.__cuda_array_interface__.copy()
        shape = cai["shape"]
        cai["shape"] = (shape[0]*shape[1],)
        self.__cuda_array_interface__ = cai


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
    def __init__(self, obj=None, *args, **kwargs):
        self._init(obj, *args, **kwargs)

    __hash__ = None

    @functools.singledispatchmethod
    def _init(self, obj, *args, **kwargs):
        if obj is None:
            if (data_type := kwargs.get("data_type")) is not None:
                kwargs.pop("data_type")
                self._init(data_type, *args, **kwargs)
                return
            elif (arrow_like := kwargs.get("arrow_like")) is not None:
                kwargs.pop("arrow_like")
                self._init(arrow_like, *args, **kwargs)
                return
        raise ValueError(f"Invalid input type {type(obj)}")

    @_init.register(DataType)
    def _(
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

    @_init.register(_ArrowLike)
    def _(self, arrow_like):
        cdef ArrowSchema* c_schema
        cdef ArrowArray* c_array
        cdef ArrowDeviceArray* c_device_array
        cdef _ArrowColumnHolder result
        cdef unique_ptr[arrow_column] c_result
        if hasattr(arrow_like, "__arrow_c_device_array__"):
            schema, array = arrow_like.__arrow_c_device_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_device_array = (
                <ArrowDeviceArray*>PyCapsule_GetPointer(array, "arrow_device_array")
            )

            result = _ArrowColumnHolder()
            with nogil:
                c_result = make_unique[arrow_column](
                    move(dereference(c_schema)), move(dereference(c_device_array))
                )
            result.col.swap(c_result)

            tmp = Column.from_column_view_of_arbitrary(result.col.get().view(), result)
            self._init(
                tmp.type(),
                tmp.size(),
                tmp.data(),
                tmp.null_mask(),
                tmp.null_count(),
                tmp.offset(),
                tmp.children(),
            )
        elif hasattr(arrow_like, "__arrow_c_array__"):
            schema, array = arrow_like.__arrow_c_array__()
            c_schema = <ArrowSchema*>PyCapsule_GetPointer(schema, "arrow_schema")
            c_array = <ArrowArray*>PyCapsule_GetPointer(array, "arrow_array")

            result = _ArrowColumnHolder()
            with nogil:
                c_result = make_unique[arrow_column](
                    move(dereference(c_schema)), move(dereference(c_array))
                )
            result.col.swap(c_result)

            tmp = Column.from_column_view_of_arbitrary(result.col.get().view(), result)
            self._init(
                tmp.type(),
                tmp.size(),
                tmp.data(),
                tmp.null_mask(),
                tmp.null_count(),
                tmp.offset(),
                tmp.children(),
            )
        else:
            raise ValueError("Invalid Arrow-like object")

    cdef column_view view(self) nogil:
        """Generate a libcudf column_view to pass to libcudf algorithms.

        This method is for pylibcudf's functions to use to generate inputs when
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef const void * data = NULL
        cdef const bitmask_type * null_mask = NULL

        if self._data is not None:
            data = <void*>self._data.ptr
        if self._mask is not None:
            null_mask = <bitmask_type*>self._mask.ptr

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
            data = <void*>self._data.ptr
        if self._mask is not None:
            null_mask = <bitmask_type*>self._mask.ptr

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
    def from_rmm_buffer(
        DeviceBuffer buff,
        DataType dtype,
        size_type size,
        list children,
    ):
        """
        Create a Column from an RMM DeviceBuffer.

        Parameters
        ----------
        buff : DeviceBuffer
            The data rmm.DeviceBuffer.
        size : size_type
            The number of rows in the column.
        dtype : DataType
            The type of the data in the buffer.
        children : list
            List of child columns.

        Notes
        -----
        To provide a mask and null count, use `Column.with_mask` after
        this method.
        """
        if plc_is_fixed_width(dtype) and len(children) != 0:
            raise ValueError("Fixed-width types must have zero children.")
        elif dtype.id() == type_id.STRING and len(children) != 1:
            raise ValueError("String columns have have 1 child column of offsets.")
        elif is_nested(dtype) and len(children) == 0:
            raise ValueError(
                "List and struct columns must have at least one child column."
            )

        cdef gpumemoryview data = gpumemoryview(buff)
        return Column(
            dtype,
            size,
            data,
            None,
            0,
            # Initial offset when capturing a C++ column is always 0.
            0,
            children,
        )

    @staticmethod
    cdef Column from_libcudf(unique_ptr[column] libcudf_col, Stream stream=None):
        """Create a Column from a libcudf column.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        cdef DataType dtype = DataType.from_libcudf(libcudf_col.get().type())
        cdef size_type size = libcudf_col.get().size()

        cdef size_type null_count = libcudf_col.get().null_count()

        cdef column_contents contents = libcudf_col.get().release()

        stream = _get_stream(stream)
        # Note that when converting to cudf Column objects we'll need to pull
        # out the base object.
        cdef gpumemoryview data = gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(contents.data), stream)
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
                    Column.from_libcudf(move(contents.children[i]), stream)
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

        cdef gpumemoryview owning_data = gpumemoryview(OwnerWithCAI.create(cv, owner))
        cdef gpumemoryview owning_mask = None
        if cv.null_count() > 0:
            owning_mask = gpumemoryview(OwnerMaskWithCAI.create(cv, owner))

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

    cpdef Scalar to_scalar(self):
        """
        Return the first value of 1-element column as a Scalar.

        Raises
        ------
        ValueError
            If the column has more than one row.

        Returns
        -------
        Scalar
            A Scalar representing the only value in the column, including nulls.
        """
        if self._size != 1:
            raise ValueError("to_scalar only works for columns of size 1")

        cdef column_view cv = self.view()
        cdef unique_ptr[scalar] result

        with nogil:
            result = get_element(cv, 0)

        return Scalar.from_libcudf(move(result))

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
            Must implement the ``__cuda_array_interface__`` protocol.

        Returns
        -------
        Column
            A Column containing the data from the CUDA array interface.

        Raises
        ------
        TypeError
            If the object does not support __cuda_array_interface__.
        ValueError
            If the object is not 1D or 2D, or is not C-contiguous.
            If the number of rows exceeds size_type limit.
        NotImplementedError
            If the object has a mask.
        """
        try:
            iface = obj.__cuda_array_interface__
        except AttributeError:
            raise TypeError("Object does not implement __cuda_array_interface__")

        if iface.get("mask") is not None:
            raise NotImplementedError("mask not yet supported")

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
            num_rows, num_cols = shape
            if num_rows < numeric_limits[size_type].max():
                offsets_col = sequence(
                    num_rows + 1,
                    Scalar.from_py(0, DataType(type_id.INT32)),
                    Scalar.from_py(num_cols, DataType(type_id.INT32)),
                )
            else:
                raise ValueError(
                    "Number of rows exceeds size_type limit for offsets column."
                )
            rav_obj = _Ravelled(obj)
            data_col = cls(
                data_type=data_type,
                size=rav_obj.__cuda_array_interface__["shape"][0],
                data=gpumemoryview(rav_obj),
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
    def from_array(cls, obj):
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
            If NumPy is not installed.

        Notes
        -----
        - 1D and 2D C-contiguous device arrays are supported.
          The data are not copied.
        - For `numpy.ndarray`, this is not yet implemented.

        Examples
        --------
        >>> import pylibcudf as plc
        >>> import cupy as cp
        >>> cp_arr = cp.array([[1,2],[3,4]])
        >>> col = plc.Column.from_array(cp_arr)
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

    cpdef uint64_t device_buffer_size(self):
        """
        The total size of the device buffers used by the Column.

        Notes
        -----
        Since Columns rely on Python memoryview-like semantics to maintain
        shared ownership of the data, the device buffers underlying this column
        might be shared between other data structures including other columns.

        Returns
        -------
        Number of bytes.
        """
        cdef uint64_t ret = 0
        if self.data() is not None:
            ret += self.data().nbytes
        if self.null_mask() is not None:
            ret += self.null_mask().nbytes
        if self.children() is not None:
            for child in self.children():
                ret += (<Column?>child).device_buffer_size()
        return ret

    def _create_nested_column_metadata(self):
        return ColumnMetadata(
            children_meta=[
                child._create_nested_column_metadata() for child in self.children()
            ]
        )

    def _to_schema(self, metadata=None):
        """Create an Arrow schema from this Column."""
        if metadata is None:
            metadata = self._create_nested_column_metadata()
        elif isinstance(metadata, str):
            metadata = ColumnMetadata(metadata)

        cdef column_metadata c_metadata = _metadata_to_libcudf(metadata)

        cdef ArrowSchema* raw_schema_ptr
        with nogil:
            raw_schema_ptr = to_arrow_schema_raw(self.view(), c_metadata)

        return PyCapsule_New(<void*>raw_schema_ptr, 'arrow_schema', _release_schema)

    def _to_host_array(self):
        cdef ArrowArray* raw_host_array_ptr
        with nogil:
            raw_host_array_ptr = to_arrow_host_raw(self.view())

        return PyCapsule_New(<void*>raw_host_array_ptr, "arrow_array", _release_array)

    def _to_device_array(self):
        cdef ArrowDeviceArray* raw_device_array_ptr
        with nogil:
            raw_device_array_ptr = to_arrow_device_raw(self.view(), self)

        return PyCapsule_New(
            <void*>raw_device_array_ptr,
            "arrow_device_array",
            _release_device_array
        )

    def __arrow_c_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Column does not support alternative schema")

        return self._to_schema(), self._to_host_array()

    def __arrow_c_device_array__(self, requested_schema=None, **kwargs):
        if requested_schema is not None:
            raise ValueError("pylibcudf.Column does not support alternative schema")

        non_default_kwargs = [
            name for name, value in kwargs.items() if value is not None
        ]
        if non_default_kwargs:
            raise NotImplementedError(
                f"Received unsupported keyword argument(s): {non_default_kwargs}"
            )

        return self._to_schema(), self._to_device_array()


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
