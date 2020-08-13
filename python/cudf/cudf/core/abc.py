# Copyright (c) 2020, NVIDIA CORPORATION.

import abc
import sys
from abc import abstractmethod

import rmm

import cudf

if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle
else:
    import pickle


class Serializable(abc.ABC):
    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, header, frames):
        pass

    def device_serialize(self):
        """Converts the object into a header and list of Buffer/memoryview
        objects for file storage or network transmission.

        Returns
        -------
            header : dictionary containing any serializable metadata
            frames : list of Buffer or memoryviews, commonly of length one

        :meta private:
        """
        header, frames = self.serialize()
        assert all(
            (type(f) in [cudf.core.buffer.Buffer, memoryview]) for f in frames
        )
        header["type-serialized"] = pickle.dumps(type(self))
        header["is-cuda"] = [
            hasattr(f, "__cuda_array_interface__") for f in frames
        ]
        header["lengths"] = [f.nbytes for f in frames]
        return header, frames

    @classmethod
    def device_deserialize(cls, header, frames):
        """Convert serialized header and frames back
        into respective Object Type

        Parameters
        ----------
        cls : class of object
        header : dict
            dictionary containing any serializable metadata
        frames : list of Buffer or memoryview objects

        Returns
        -------
        Deserialized Object of type cls extracted
        from frames and header

        :meta private:
        """
        typ = pickle.loads(header["type-serialized"])
        frames = [
            cudf.core.buffer.Buffer(f) if c else memoryview(f)
            for c, f in zip(header["is-cuda"], frames)
        ]
        assert all(
            (type(f._owner) is rmm.DeviceBuffer)
            if c
            else (type(f) is memoryview)
            for c, f in zip(header["is-cuda"], frames)
        )
        obj = typ.deserialize(header, frames)

        return obj

    def host_serialize(self):
        """Converts the object into a header and list of memoryview
        objects for file storage or network transmission.

        Returns
        -------
            header : dictionary containing any serializable metadata
            frames : list of memoryviews, commonly of length one

        :meta private:
        """
        header, frames = self.device_serialize()
        header["writeable"] = len(frames) * (None,)
        frames = [
            f.to_host_array().data if c else memoryview(f)
            for c, f in zip(header["is-cuda"], frames)
        ]
        return header, frames

    @classmethod
    def host_deserialize(cls, header, frames):
        """Convert serialized header and frames back
        into respective Object Type

        Parameters
        ----------
        cls : class of object
        header : dict
            dictionary containing any serializable metadata
        frames : list of memoryview objects

        Returns
        -------
        Deserialized Object of type cls extracted
        from frames and header

        :meta private:
        """
        frames = [
            rmm.DeviceBuffer.to_device(f) if c else f
            for c, f in zip(header["is-cuda"], map(memoryview, frames))
        ]
        obj = cls.device_deserialize(header, frames)
        return obj

    def __reduce_ex__(self, protocol):
        header, frames = self.host_serialize()
        frames = [f.obj for f in frames]
        return self.host_deserialize, (header, frames)
