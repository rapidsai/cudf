# Copyright (c) 2020, NVIDIA CORPORATION.

import abc
import pickle
from abc import abstractmethod

import rmm

import cudf


class Serializable(abc.ABC):
    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, header, frames):
        pass

    def device_serialize(self):
        header, frames = self.serialize()
        assert all((type(f) is cudf.core.buffer.Buffer) for f in frames)
        header["type-serialized"] = pickle.dumps(type(self))
        header["lengths"] = [f.nbytes for f in frames]
        return header, frames

    @classmethod
    def device_deserialize(cls, header, frames):
        typ = pickle.loads(header["type-serialized"])
        frames = [cudf.core.buffer.Buffer(f) for f in frames]
        obj = typ.deserialize(header, frames)

        return obj

    def host_serialize(self):
        header, frames = self.device_serialize()
        frames = [f.to_host_array().data for f in frames]
        return header, frames

    @classmethod
    def host_deserialize(cls, header, frames):
        frames = [rmm.DeviceBuffer.to_device(memoryview(f)) for f in frames]
        obj = cls.device_deserialize(header, frames)
        return obj

    def __reduce_ex__(self, protocol):
        header, frames = self.host_serialize()
        frames = [f.obj for f in frames]
        return self.host_deserialize, (header, frames)
