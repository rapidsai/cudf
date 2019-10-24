class Buffer:
    def __init__(self, ptr, size, owner=None):
        self.ptr = ptr
        self.size = size
        self._owner = owner
