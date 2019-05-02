import logging
import json
from contextlib import contextmanager
from collections import namedtuple, OrderedDict
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from cudf.utils.utils import mask_dtype, mask_bitsize
from cudf.dataframe import Series
from libgdf_cffi import ffi, libgdf


_logger = logging.getLogger(__name__)

_BufferDesc = namedtuple("_BufferDesc", "offset,length")
_NodeDesc = namedtuple(
    "_NodeDesc",
    'name,length,null_count,null_buffer,data_buffer,dtype,schema'
    )


class MetadataParsingError(ValueError):
    pass


# TODO: can this be replaced with calls to rmm.device_array_from_ptr()?
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
    elif name in ('INT', 'INT8', 'INT16', 'INT32', 'INT64'):
        ret = getattr(np, 'int{:d}'.format(bitwidth))
    elif name == 'DICTIONARY':
        ret = getattr(np, 'int{:d}'.format(bitwidth))
    else:
        fmt = "unsupported type {} {}-bits"
        raise NotImplementedError(fmt.format(name, bitwidth))
    return np.dtype(ret)


@contextmanager
def _open_parser(schema_ptr, schema_len):
    "context to destroy the parser"
    _logger.debug('open IPCParser')
    ipcparser = libgdf.gdf_ipc_parser_open(schema_ptr, schema_len)
    yield ipcparser
    _logger.debug('close IPCParser')
    libgdf.gdf_ipc_parser_close(ipcparser)


def _check_error(ipcparser):
    if libgdf.gdf_ipc_parser_failed(ipcparser):
        raw_error = libgdf.gdf_ipc_parser_get_error(ipcparser)
        error = ffi.string(raw_error).decode()
        _logger.error('IPCParser failed: %s', error)
        raise MetadataParsingError(error)


def _load_json(jsonraw):
    jsontext = ffi.string(jsonraw).decode()
    return json.loads(jsontext)


class GpuArrowNodeReader(object):
    def __init__(self, schema, gpu_data, desc):
        self._schema = schema
        self._gpu_data = gpu_data
        self._desc = desc

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
        return 'dictionary' in self.field_schema

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
        bits = mask_bitsize
        itemsize = mask_dtype.itemsize
        end = ((self._desc.length + bits - 1) // bits) * itemsize
        return gpu_view_as(self.null_raw[:end], dtype=mask_dtype)

    def make_series(self):
        """Make a Series object out of this node
        """
        if self.is_dictionary:
            sr = self._make_dictionary_series()
        else:
            sr = Series(self.data)

        # set nullmask
        if self.null_count:
            sr = sr.set_mask(self.null, null_count=self.null_count)

        return sr

    def _make_dictionary_series(self):
        """Make a dictionary-encoded series from this node
        """
        assert self.is_dictionary
        # create dictionary-encoded column
        dict_meta = self.field_schema['dictionary']
        dictid = dict_meta['id']   # start from 1
        if dict_meta['indexType']['name'] != 'int':
            msg = 'non integer type index for dictionary'
            raise MetadataParsingError(msg)
        ordered = dict_meta['isOrdered']
        # find dictionary
        for dictionary in self.schema['dictionaries']:
            if dictionary['id'] == dictid:
                break
        categories = dictionary['data']['columns'][0]['DATA']
        # make dummy categorical
        cat = pd.Categorical([], categories=categories, ordered=ordered)
        # make the series
        return Series.from_categorical(cat, codes=self.data)


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
            dc[node.name] = node.make_series()
        return dc

    #
    # Private API
    #
    def _open(self):
        schema, layoutinfo, dataptr = self._parse_metdata()
        for fielddesc, layout in zip(schema['schema']['fields'], layoutinfo):
            _logger.debug('reading data from libgdf IPCParser')
            nodedesc = _NodeDesc(
                name=layout['name'],
                length=layout['length'],
                null_count=layout['null_count'],
                null_buffer=_BufferDesc(**layout['null_buffer']),
                data_buffer=_BufferDesc(**layout['data_buffer']),
                dtype=_schema_to_dtype(**layout['dtype']),
                schema=fielddesc,
                )
            node = GpuArrowNodeReader(schema=schema,
                                      gpu_data=dataptr,
                                      desc=nodedesc)
            self._nodes.append(node)

    def _parse_metdata(self):
        "Parse the metadata in the IPC handle"

        # get void* from the gpu array
        schema_ptr = ffi.cast("void*", self._schema_data.ctypes.data)

        # parse schema
        with _open_parser(schema_ptr, len(self._schema_data)) as ipcparser:
            # check for failure in parseing the schema
            _check_error(ipcparser)

            gpu_addr = self._gpu_data.device_ctypes_pointer.value
            gpu_ptr = ffi.cast("void*", gpu_addr)
            libgdf.gdf_ipc_parser_open_recordbatches(ipcparser, gpu_ptr,
                                                     self._gpu_data.size)
            # check for failure in parsing the recordbatches
            _check_error(ipcparser)
            # get schema as json
            _logger.debug('IPCParser get metadata as json')
            schemadct = _load_json(
                libgdf.gdf_ipc_parser_get_schema_json(ipcparser))
            layoutdct = _load_json(
                libgdf.gdf_ipc_parser_get_layout_json(ipcparser))

            # get data offset
            _logger.debug('IPCParser data region offset')
            dataoffset = libgdf.gdf_ipc_parser_get_data_offset(ipcparser)
            dataoffset = int(ffi.cast('uint64_t', dataoffset))
            dataptr = self._gpu_data[dataoffset:]

        return schemadct, layoutdct, dataptr
