import logging
import warnings
from collections import namedtuple, Sequence

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from org.apache.arrow.flatbuf import RecordBatch, Message, MessageHeader


_logger = logging.getLogger(__name__)

_MessageInfo = namedtuple('_MessageInfo', 'header,type,body_length')

_BufferDesc = namedtuple("_BufferDesc", "offset,length")
_NodeDesc = namedtuple("_NodeDesc", 'length,null_count,null_buffer,data_buffer')


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


class GpuArrowNodeReader(object):
    def __init__(self, gpu_data, desc):
        self._gpu_data = gpu_data
        self._desc = desc
        if desc.null_count != 0:
            raise NotImplementedError('null-mask is needed')

    @property
    def data(self):
        "Accessor for the data buffer as a device array"
        size = self._desc.data_buffer.length
        start = self._desc.data_buffer.offset
        stop = start + size
        ary = self._gpu_data[start:stop]
        if ary.size != size:
            raise ValueError('data size mismatch')
        return ary

    def data_as(self, dtype):
        """
        Return the data as the given dtype with the padding bytes truncated.
        """
        dtype = np.dtype(dtype)
        end = self._desc.length * dtype.itemsize
        return gpu_view_as(self.data[:end], dtype=dtype)


class GpuArrowReader(Sequence):
    def __init__(self, gpu_data):
        loggername = '{}@{:08x}'.format(self.__class__.__name__, id(self))
        self._logger = _logger.getChild(loggername)
        self._gpu_data = gpu_data
        self._readidx = 0
        self._nodes = []

        self._open()

    #
    # Public API
    #

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, idx):
        return self._nodes[idx]

    #
    # Metadata parsing
    #

    def _open(self):
        self.read_schema()
        self.read_recordbatch()

    def read_schema(self):
        # Read schema
        self._logger.debug("reading schema")
        size = self.read_msg_size()
        self._get(size)  # skip schema body

    def read_recordbatch(self):
        # Read RecordBatch
        self._logger.debug("reading recordbatch")
        size = self.read_msg_size()
        # Read message header
        msg = self.read_msg_header(size)
        body = self._get(msg.body_length)
        for node in self.parse_record_batch(msg):
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

    def read_int32(self):
        int32 = np.dtype(np.int32)
        nbytes = int32.itemsize
        data = self._get(nbytes)
        return gpu_view_as(data, dtype=int32)[0]

    def read_msg_size(self):
        size = self.read_int32()
        if size < 0:
            msg = 'invalid message size: ({}) < 0'.format(size)
            raise MetadataParsingError(msg)
        return size

    def read_msg_header(self, size):
        buffer = self._get(size).copy_to_host()
        header = self.parse_msg_header(buffer)
        return header

    def parse_msg_header(self, buffer):
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

    def parse_record_batch(self, msg):
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
            node = rb.Nodes(i)
            node_buffers = []
            for j in range(buffer_per_node):
                buf = rb.Buffers(i * buffer_per_node + j)
                # This is a less important check, so just warn
                if buf.Page() != -1:
                    warnings.warn('buf.Page() != -1; metadata format changed')
                bufdesc = _BufferDesc(offset=buf.Offset(), length=buf.Length())
                node_buffers.append(bufdesc)
            null_buffer, data_buffer = node_buffers
            desc = _NodeDesc(length=node.Length(), null_count=node.NullCount(),
                             null_buffer=null_buffer, data_buffer=data_buffer)
            self._logger.debug("got node: %s", desc)
            nodes.append(desc)

        self._logger.debug('end parsing record batch')
        return nodes

