from cudf._lib.nvtx._lib.lib cimport *


def initialize():
    nvtxInitialize(NULL)


cdef class EventAttributes:
    cdef dict __dict__
    cdef nvtxEventAttributes_t c_obj

    def __init__(self, message=None, color=None):
        self._message = message.encode("ascii")
        self._color = color
        self.c_obj = nvtxEventAttributes_t(0)
        self.c_obj.version = NVTX_VERSION
        self.c_obj.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
        self.c_obj.colorType = NVTX_COLOR_ARGB
        self.c_obj.color = color
        self.c_obj.messageType = NVTX_MESSAGE_TYPE_ASCII
        self.c_obj.message.ascii = <const char*> self._message

    @property
    def message(self):
        return self._message


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


def range_push(EventAttributes attributes, DomainHandle domain):
    nvtxDomainRangePushEx(domain.c_obj, &attributes.c_obj)


def range_pop():
    nvtxRangePop()


def range_start(EventAttributes attributes, DomainHandle domain):
    return RangeId(nvtxDomainRangeStartEx(domain.c_obj, &attributes.c_obj))


def range_end(RangeId r_id, DomainHandle domain):
    nvtxDomainRangeEnd(domain.c_obj, r_id.c_obj)
