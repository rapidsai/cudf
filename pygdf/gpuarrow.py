import logging
import warnings
from collections import namedtuple, Sequence, OrderedDict

import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from org.apache.arrow.flatbuf import (RecordBatch, Message, MessageHeader,
                                      Schema, Type, VectorType)


from .utils import mask_dtype
from .dataframe import Series


_logger = logging.getLogger(__name__)

_MessageInfo = namedtuple('_MessageInfo', 'header,type,body_length')

_BufferDesc = namedtuple("_BufferDesc", "offset,length")
_NodeDesc = namedtuple("_NodeDesc", 'name,length,null_count,null_buffer,data_buffer,dtype')
_LayoutDesc = namedtuple("_LayoutDesc", "bitwidth,vectortype")
_FieldDesc = namedtuple("_FieldDesc", "name,type,layouts")


class MetadataParsingError(ValueError):
    pass


def gpu_view_as(arr, dtype, shape=None, strides=None):
    dtype = np.dtype(dtype)
    if strides is None:
        strides = (arr.strides
                   if arr.dtype == dtype
                   else dtype.itemsize)
    if shape is None:
        shape = (arr.shape
                 if arr.dtype == dtype
                 else arr.size // dtype.itemsize)
    return DeviceNDArray(shape=shape, strides=strides, dtype=dtype,
                         gpu_data=arr.gpu_data)


typenames = '''Null Int FloatingPoint Binary Utf8 Bool Decimal Date Time
Timestamp Interval List Struct_ Union FixedSizeBinary'''.split()


def _find_name_in_enum(value, typ, names):
    for k in names:
        if getattr(typ, k) == value:
            return k
    else:
        raise ValueError(value, typ)


def _determine_layout_type(value):
    names = ['OFFSET', 'DATA', 'VALIDITY', 'TYPE']
    return _find_name_in_enum(value, typ=VectorType.VectorType, names=names)


def _determine_schema_type(value):
    return _find_name_in_enum(value, typ=Type.Type, names=typenames)


def _schema_to_dtype(datatype, bitwidth):
    if datatype == 'FloatingPoint':
        ret = getattr(np, 'float{:d}'.format(bitwidth))
    elif datatype == 'Int':
        ret = getattr(np, 'int{:d}'.format(bitwidth))
    else:
        fmt = "unsupported type {} {}-bits"
        raise NotImplementedError(fmt.format(datatype, bitwidth))
    return np.dtype(ret)


class GpuArrowNodeReader(object):
    def __init__(self, gpu_data, desc):
        self._gpu_data = gpu_data
        self._desc = desc
        if self._desc.null_count:
            if self._desc.length != self._desc.null_count:
                msg = "unexpected self._desc.length != self._desc.null_count"
                raise NotImplementedError(msg)

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
        size = self._desc.data_buffer.length
        start = self._desc.data_buffer.offset
        stop = start + size
        ary = self._gpu_data[start:stop]
        if ary.size != size:
            raise ValueError('data size mismatch')
        return ary

    @property
    def data(self):
        """
        Return the data as the expected dtype
        and with the padding bytes truncated.
        """
        end = self._desc.length * self.dtype.itemsize
        return gpu_view_as(self.data_raw[:end], dtype=self.dtype)

    @property
    def null_raw(self):
        "Accessor for the null buffer as a device array"
        size = self._desc.null_buffer.length
        start = self._desc.null_buffer.offset
        stop = start + size
        ary = self._gpu_data[start:stop]
        if ary.size != size:
            raise ValueError('data size mismatch')
        return ary

    @property
    def null(self):
        """
        Return the null mask with the padding bytes truncated.
        """
        end = (self._desc.length // 8) * mask_dtype.itemsize
        return gpu_view_as(self.null_raw[:end], dtype=mask_dtype)


class GpuArrowReader(Sequence):
    def __init__(self, gpu_data):
        loggername = '{}@{:08x}'.format(self.__class__.__name__, id(self))
        self._logger = _logger.getChild(loggername)
        self._gpu_data = gpu_data
        self._readidx = 0
        self._fields = []
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
            if node.null_count:
                sr = Series.from_masked_array(data=node.data,
                                              mask=node.null,
                                              null_count=node.null_count)

            else:
                sr = Series.from_array(node.data)
            dc[node.name] = sr
        return dc

    #
    # Metadata parsing
    #

    def _open(self):
        self._read_schema()
        self._read_recordbatch()

    def _read_schema(self):
        # Read schema
        self._logger.debug("reading schema")
        size = self._read_msg_size()
        schema_buf = self._get(size).copy_to_host()
        header = self._parse_msg_header(schema_buf)
        if header.body_length > 0:
            raise MetadataParsingError("schema should not have body")
        fds = self._parse_schema(header)
        self._fields.extend(fds)

    def _read_recordbatch(self):
        # Read RecordBatch
        self._logger.debug("reading recordbatch")
        size = self._read_msg_size()
        # Read message header
        msg = self._read_msg_header(size)
        body = self._get(msg.body_length)
        for i, node in enumerate(self._parse_record_batch(msg)):
            self._nodes.append(GpuArrowNodeReader(gpu_data=body, desc=node))

    def _get(self, size):
        "Get the offseted buffer"
        self._logger.debug('offset=%d size=%d end=%d',
                           self._readidx, size, self._gpu_data.size)
        start = self._readidx
        stop = start + size
        ret = self._gpu_data[start:stop]
        self._readidx += size
        return ret

    def _read_int32(self):
        int32 = np.dtype(np.int32)
        nbytes = int32.itemsize
        data = self._get(nbytes)
        return gpu_view_as(data, dtype=int32)[0]

    def _read_msg_size(self):
        size = self._read_int32()
        if size < 0:
            msg = 'invalid message size: ({}) < 0'.format(size)
            raise MetadataParsingError(msg)
        return size

    def _read_msg_header(self, size):
        buffer = self._get(size).copy_to_host()
        header = self._parse_msg_header(buffer)
        return header

    def _parse_msg_header(self, buffer):
        self._logger.debug('parsing message header')
        msg = Message.Message.GetRootAsMessage(buffer, 0)
        body_size = msg.BodyLength()
        self._logger.debug('header: body size = %s', body_size)
        version = msg.Version()
        self._logger.debug('header: version = %s', version)
        header_type = msg.HeaderType()
        self._logger.debug('header: type = %s', header_type)
        self._logger.debug('end parsing message header')
        return _MessageInfo(header=msg.Header(), type=header_type,
                            body_length=body_size)

    def _parse_schema(self, msg):
        if msg.type != MessageHeader.MessageHeader.Schema:
            errmsg = 'expecting schema type'
            raise MetadataParsingError(errmsg)

        self._logger.debug('parsing schema')
        schema = Schema.Schema()
        schema.Init(msg.header.Bytes, msg.header.Pos)

        fields = []
        for i in range(schema.FieldsLength()):
            field = schema.Fields(i)
            name = field.Name().decode('utf8')
            self._logger.debug('field %d: name=%r', i, name)
            fieldtype = _determine_schema_type(field.TypeType())
            self._logger.debug('field %d: type=%s', i, fieldtype)

            layouts = []
            for j in range(field.LayoutLength()):
                layout = field.Layout(j)
                bitwidth = layout.BitWidth()
                self._logger.debug(' layout %d: bitwidth=%s', j, bitwidth)
                vectype = _determine_layout_type(layout.Type())
                self._logger.debug(' layout %d: vectortype=%s', j, vectype)
                layouts.append(_LayoutDesc(bitwidth=bitwidth,
                                           vectortype=vectype))

            fields.append(_FieldDesc(name=name, type=fieldtype,
                                     layouts=layouts))

        self._logger.debug('end parsing schema')
        return fields

    def _parse_record_batch(self, msg):
        if msg.type != MessageHeader.MessageHeader.RecordBatch:
            errmsg = 'expecting record batch type'
            raise MetadataParsingError(errmsg)

        self._logger.debug('parsing record batch')
        rb = RecordBatch.RecordBatch()
        rb.Init(msg.header.Bytes, msg.header.Pos)

        # Parse nodes. Expects two buffers per node for null-mask and data
        node_ct = rb.NodesLength()
        buffer_ct = rb.BuffersLength()
        buffer_per_node = 2
        if node_ct * buffer_per_node != buffer_ct:
            raise MetadataParsingError('more then 2 buffers per node?!')

        nodes = []
        for i in range(node_ct):
            fd = self._fields[i]
            node = rb.Nodes(i)
            node_buffers = {}
            for j in range(buffer_per_node):
                buf = rb.Buffers(i * buffer_per_node + j)
                # This is a less important check, so just warn
                if buf.Page() != -1:
                    warnings.warn('buf.Page() != -1; metadata format changed')

                layout = fd.layouts[j]
                bufdesc = _BufferDesc(offset=buf.Offset(), length=buf.Length())
                assert layout.vectortype not in node_buffers
                node_buffers[layout.vectortype] = bufdesc

            dtype = _schema_to_dtype(fd.type, fd.layouts[j].bitwidth)
            desc = _NodeDesc(name=fd.name,
                             length=node.Length(),
                             null_count=node.NullCount(),
                             null_buffer=node_buffers['VALIDITY'],
                             data_buffer=node_buffers['DATA'],
                             dtype=dtype)
            self._logger.debug("got node: %s", desc)
            nodes.append(desc)

        self._logger.debug('end parsing record batch')
        return nodes

