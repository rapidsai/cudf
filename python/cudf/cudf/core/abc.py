# Copyright (c) 2020-2021, NVIDIA CORPORATION.
"""Common abstract base classes for cudf."""

import sys

import rmm

import cudf

if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle  # type: ignore
else:
    import pickle  # type: ignore


class Serializable:
    """A serializable object composed of device memory buffers.

    This base class defines a standard serialization protocol for objects
    encapsulating device memory buffers. Serialization proceeds by copying
    device data onto the host, then returning it along with suitable metadata
    for reconstruction of the object. Deserialization performs the reverse
    process, copying the serialized data from the host to new device buffers.
    Subclasses must define the abstract methods :meth:`~.serialize` and
    :meth:`~.deserialize`. The former defines the conversion of the object
    into a representative collection of metadata and data buffers, while the
    latter converts back from that representation into an equivalent object.
    """

    def serialize(self):
        """Generate an equivalent serializable representation of an object.

        Subclasses must implement this method to define how the attributes of
        the object are converted into a serializable representation. A common
        solution is to construct a list containing device buffer attributes in
        a well-defined order that can be reinterpreted upon deserialization,
        then place all other lightweight attributes into the metadata
        dictionary.

        Returns
        -------
        Tuple[Dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing the device data buffers
            or memoryviews of the object.

        :meta private:
        """
        raise NotImplementedError(
            "Subclasses of Serializable must implement serialize"
        )

    @classmethod
    def deserialize(cls, header, frames):
        """Generate an object from a serialized representation.

        Subclasses must implement this method to define how objects of that
        class can be constructed from a serialized representation generalized
        by :meth:`serialize`.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffers or memoryviews that the object should contain.

        Returns
        -------
        Serializable
            A new instance of `cls` (a subclass of `Serializable`) equivalent
            to the instance that was serialized to produce the header and
            frames.

        :meta private:
        """
        raise NotImplementedError(
            "Subclasses of Serializable must implement deserialize"
        )

    def device_serialize(self):
        """Serialize data and metadata associated with device memory.

        Returns
        -------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffers or memoryviews that the object should contain.

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
        """Perform device-side deserialization tasks.

        The primary purpose of this method is the creation of device memory
        buffers from host buffers where necessary.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffers or memoryviews that the object should contain.

        Returns
        -------
        Serializable
            A new instance of `cls` (a subclass of `Serializable`) equivalent
            to the instance that was serialized to produce the header and
            frames.

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
        """Serialize data and metadata associated with host memory.

        Returns
        -------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffers or memoryviews that the object should contain.

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
        """Perform device-side deserialization tasks.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffers or memoryviews that the object should contain.

        Returns
        -------
        Serializable
            A new instance of `cls` (a subclass of `Serializable`) equivalent
            to the instance that was serialized to produce the header and
            frames.

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
