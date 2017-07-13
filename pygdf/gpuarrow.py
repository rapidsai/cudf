import logging
import json
from contextlib import contextmanager
from collections import namedtuple, Sequence, OrderedDict

import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from .utils import mask_dtype
from .dataframe import Series


_logger = logging.getLogger(__name__)

_BufferDesc = namedtuple("_BufferDesc", "offset,length")
_NodeDesc = namedtuple(
    "_NodeDesc",
    'name,length,null_count,null_buffer,data_buffer,dtype'
    )


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


def _schema_to_dtype(name, bitwidth):
    if name in ('DOUBLE', 'FLOAT'):
        ret = getattr(np, 'float{:d}'.format(bitwidth))
    elif name in ('INT', 'INT32', 'INT64'):
        ret = getattr(np, 'int{:d}'.format(bitwidth))
    elif name == 'DICTIONARY':
        ret = getattr(np, 'int{:d}'.format(bitwidth))
    else:
        fmt = "unsupported type {} {}-bits"
        raise NotImplementedError(fmt.format(name, bitwidth))
    return np.dtype(ret)


class GpuArrowNodeReader(object):
    def __init__(self, gpu_data, desc):
        self._gpu_data = gpu_data
        self._desc = desc

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
    def __init__(self, schema_data, gpu_data):
        loggername = '{}@{:08x}'.format(self.__class__.__name__, id(self))
        self._logger = _logger.getChild(loggername)
        self._schema_data = schema_data
        self._gpu_data = gpu_data
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
    # Private API
    #

    def _open(self):
        schema, layoutinfo, dataptr = self._parse_metdata()
        from pprint import pprint
        pprint(schema)
        pprint(layoutinfo)
        for fielddesc, layout in zip(schema['schema']['fields'], layoutinfo):
            _logger.debug('reading data from libgdf IPCParser')
            nodedesc = _NodeDesc(
                name=layout['name'],
                length=layout['length'],
                null_count=layout['null_count'],
                null_buffer=_BufferDesc(**layout['null_buffer']),
                data_buffer=_BufferDesc(**layout['data_buffer']),
                dtype=_schema_to_dtype(**layout['dtype']),
                )
            node = GpuArrowNodeReader(gpu_data=dataptr,
                                      desc=nodedesc)
            self._nodes.append(node)

    def _parse_metdata(self):
        "Parse the metadata in the IPC handle"
        from libgdf_cffi import ffi, libgdf

        @contextmanager
        def open_parser(schema_ptr, schema_len):
            "context to destroy the parser"
            _logger.debug('open IPCParser')
            ipcparser = libgdf.gdf_ipc_parser_open(schema_ptr, schema_len)
            yield ipcparser
            _logger.debug('close IPCParser')
            libgdf.gdf_ipc_parser_close(ipcparser)

        def check_error(ipcparser):
            if libgdf.gdf_ipc_parser_failed(ipcparser):
                raw_error = libgdf.gdf_ipc_parser_get_error(ipcparser)
                error = ffi.string(raw_error).decode()
                _logger.error('IPCParser failed: %s', error)
                raise MetadataParsingError(error)

        def load_json(jsonraw):
            jsontext = ffi.string(jsonraw).decode()
            return json.loads(jsontext)

        # get void* from the gpu array
        schema_ptr = ffi.cast("void*", self._schema_data.ctypes.data)

        # parse schema
        with open_parser(schema_ptr, len(self._schema_data)) as ipcparser:
            # check for failure in parseing the schema
            check_error(ipcparser)

            gpu_ptr = ffi.cast("void*", self._gpu_data.device_ctypes_pointer.value)
            libgdf.gdf_ipc_parser_open_recordbatches(ipcparser, gpu_ptr,
                                                     self._gpu_data.size)
            # check for failure in parsing the recordbatches
            check_error(ipcparser)
            # get schema as json
            _logger.debug('IPCParser get metadata as json')
            schemadct = load_json(
                libgdf.gdf_ipc_parser_get_schema_json(ipcparser))
            layoutdct = load_json(
                libgdf.gdf_ipc_parser_get_layout_json(ipcparser))

            # get data offset
            _logger.debug('IPCParser data region offset')
            dataoffset = libgdf.gdf_ipc_parser_get_data_offset(ipcparser)
            dataoffset = int(ffi.cast('uint64_t', dataoffset))
            dataptr = self._gpu_data[dataoffset:]

        return schemadct, layoutdct, dataptr
