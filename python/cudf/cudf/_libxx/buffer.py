class Buffer:
    def __init__(self, ptr=None, size=None, owner=None):
        self.ptr = ptr
        self.size = size
        self._owner = owner

    @classmethod
    def from_array_like(cls, data):
        if hasattr(data, "__cuda_array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            return cls.__new__(cls, ptr, size, owner=data)
        elif isinstance(data, "__array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = DeviceBuffer(ptr, size)
            return cls.__new__(cls, dbuf.ptr, dbuf.size, owner=dbuf)
        elif isinstance(data, DeviceBuffer):
            return cls.__new__(cls, data.ptr, data.size, owner=data)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    itemsize = np.dtype(desc["typestr"]).itemsize
    size = functools.reduce(operator.mul, array_interface["shape"])
    return ptr, size
