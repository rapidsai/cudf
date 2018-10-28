import logging
from collections import namedtuple, OrderedDict
from collections.abc import Sequence
import numpy as np
import pandas as pd
import pyarrow as pa

from .utils import mask_dtype, mask_bitsize
from .cudaarray import CudaNDArray
from .dataframe import Series

_logger = logging.getLogger(__name__)

_BufferDesc = namedtuple("_BufferDesc", "offset,length")
_NodeDesc = namedtuple(
    "_NodeDesc",
    'name,length,null_count,null_buffer,data_buffer,dtype,schema'
    )


def cuda_view_as(arr, dtype, shape=None, strides=None):
    dtype = np.dtype(dtype)
    if strides is None:
        strides = (arr.strides
                   if arr.dtype == dtype
                   else dtype.itemsize)
    if shape is None:
        shape = (arr.shape
                 if arr.dtype == dtype
                 else arr.size // dtype.itemsize)
    return CudaNDArray(shape=shape, strides=strides, dtype=dtype,
                       cuda_data=arr.cuda_data)


class MetadataParsingError(ValueError):
    pass


class CudaArrowNodeReader(object):

    def __init__(self, schema, cuda_data, desc):
        self._schema = schema
        self._desc = desc
        self._cuda_data = cuda_data  # CudaBuffer

    @property
    def schema(self):
        """Access to the schema of the result set
        """
        return self._schema

    @property
    def field_schema(self):
        """Access to the schema of this field
        """
        return self._desc.schema

    @property
    def is_dictionary(self):
        return isinstance(self.field_schema.type, pa.lib.DictionaryType)

    @property
    def null_count(self):
        return self._desc.null_count

    @property
    def dtype(self):
        return self._desc.dtype

    @property
    def name(self):
        return self._desc.name

    @property
    def data_raw(self):
        "Accessor for the data buffer as a device array"
        size = self._desc.length
        start = self._desc.data_buffer.offset
        ary = self._cuda_data.slice(start, size)  # CudaBuffer
        if ary.size != size:
            raise ValueError('data size mismatch')
        return CudaNDArray(size, cuda_data=ary)

    @property
    def data(self):
        """
        Return the data as the expected dtype
        and with the padding bytes truncated.
        """
        end = self._desc.length * self.dtype.itemsize
        return cuda_view_as(self.data_raw[:end], dtype=self.dtype)

    @property
    def null_raw(self):
        "Accessor for the null buffer as a device array"
        size = self._desc.null_buffer.length
        start = self._desc.null_buffer.offset
        ary = self._cuda_data.slice(start, size)  # CudaBuffer
        if ary.size != size:
            raise ValueError('data size mismatch')
        return CudaNDArray(size, cuda_data=ary)

    @property
    def null(self):
        """
        Return the null mask with the padding bytes truncated.
        """
        bits = mask_bitsize
        itemsize = mask_dtype.itemsize
        end = ((self._desc.length + bits - 1) // bits) * itemsize
        return cuda_view_as(self.null_raw[:end], dtype=mask_dtype)

    def make_series(self):
        """Make a Series object out of this node
        """
        if self.is_dictionary:
            itype = self.field_schema.type.index_type
            if not str(itype).startswith('int'):
                raise TypeError('non integer type index for'
                                ' dictionary: %s' % (itype))
            categories = self.field_schema.type.dictionary
            ordered = self.field_schema.type.ordered
            cat = pd.Categorical([], categories=categories, ordered=ordered)
            sr = Series.from_categorical(cat, codes=self.data)
        else:
            sr = Series(self.data)
        # set nullmask
        if self.null_count:
            sr = sr.set_mask(self.null, null_count=self.null_count)
        return sr


class CudaArrowReader(Sequence):

    def __init__(self, schema_data, cuda_data):
        loggername = '{}@{:08x}'.format(self.__class__.__name__, id(self))
        self._logger = _logger.getChild(loggername)
        self._schema_data = schema_data
        self._cuda_data = cuda_data
        self._nodes = []
        self._open()

    #
    # Public API
    #

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, idx):
        return self._nodes[idx]

    def to_dict(self):
        """
        Return a dictionary of Series object
        """
        dc = OrderedDict()
        for node in self:
            dc[node.name] = node.make_series()
        return dc

    #
    # Private API
    #
    def _open(self):
        sbuf = pa.py_buffer(self._schema_data)
        schema = pa.ipc.read_schema(sbuf)

        # Ideally, to access device data, one should use:
        #
        # rb = pa.cuda.read_record_batch(schema, self._cuda_data)
        #
        # but currently the returned RecordBatch contains device
        # pointers. With the RecordBatch implementation, one cannot
        # access the device data from Python. It is an open topic, see
        # ARROW-2447.

        cbuf = self._cuda_data.cuda_data
        gpu_address = cbuf.address
        cbatch = pa.cuda.read_record_batch(cbuf, schema)
        rows = cbatch.num_rows
        names = schema.names

        def _to_dtype(typ):
            if isinstance(typ, pa.lib.DictionaryType):
                return np.dtype('int%s' % (typ.bit_width))
            return np.dtype(typ.to_pandas_dtype())

        for i, col in enumerate(cbatch):
            null_buf, data_buf = col.buffers()
            if null_buf is None:
                null_offset = 0
                null_size = 0
            else:
                null_offset = null_buf.address - gpu_address
                null_size = null_buf.size
            data_offset = data_buf.address - gpu_address
            data_size = data_buf.size

            col_size = col.type.bit_width // 8 * rows
            nodedesc = _NodeDesc(
                name=names[i],
                length=col_size,
                null_count=col.null_count,
                null_buffer=_BufferDesc(offset=null_offset,
                                        length=null_size),
                data_buffer=_BufferDesc(offset=data_offset,
                                        length=data_size),
                dtype=_to_dtype(col.type),
                schema=schema[i],
                )
            node = CudaArrowNodeReader(schema=schema,
                                       cuda_data=cbuf, desc=nodedesc)
            self._nodes.append(node)
