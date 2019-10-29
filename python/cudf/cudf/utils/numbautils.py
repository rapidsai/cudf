class PatchedNumbaDeviceArray(object):
    def __init__(self, numba_ary):
        self.parent = numba_ary

    def __getattr__(self, name):
        if name != "__cuda_array_interface__":
            return getattr(self.parent, name)
        else:
            rtn = self.parent.__cuda_array_interface__
            if rtn.get("strides") is None:
                rtn.pop("strides")
            return rtn
