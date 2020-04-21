# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.nvtx._lib.lib cimport *
from cudf._lib.nvtx.colors import color_to_hex
from cudf._lib.nvtx.utils.cached import CachedInstanceMeta


def initialize():
    nvtxInitialize(NULL)


cdef class EventAttributes:
    cdef dict __dict__
    cdef nvtxEventAttributes_t c_obj

    def __init__(self, message=None, color="blue"):
        if message is None:
            message = ""
        self._message = message.encode("ascii")
        self._color = color_to_hex(color)
        self.c_obj = nvtxEventAttributes_t(0)
        self.c_obj.version = NVTX_VERSION
        self.c_obj.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
        self.c_obj.colorType = NVTX_COLOR_ARGB
        self.c_obj.color = self._color
        self.c_obj.messageType = NVTX_MESSAGE_TYPE_ASCII
        self.c_obj.message.ascii = <const char*> self._message

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = value.encode("ascii")
        self.c_obj.message.ascii = <const char*> self._message


cdef class DomainHandle:
    cdef dict __dict__
    cdef nvtxDomainHandle_t c_obj

    def __init__(self, name=None):
        if name is not None:
            self._name = name.encode("ascii")
            self.c_obj = nvtxDomainCreateA(self._name)
        else:
            self._name = None
            self.c_obj = NULL

    def __dealloc__(self):
        nvtxDomainDestroy(self.c_obj)


cdef class RangeId:
    cdef nvtxRangeId_t c_obj

    def __cinit__(self, uint64_t range_id):
        self.c_obj = range_id


class Domain(metaclass=CachedInstanceMeta):
    def __init__(self, name=None):
        self.name = name
        self.handle = DomainHandle(name)


def push_range(EventAttributes attributes, DomainHandle domain):
    nvtxDomainRangePushEx(domain.c_obj, &attributes.c_obj)


def pop_range(DomainHandle domain):
    nvtxDomainRangePop(domain.c_obj)
