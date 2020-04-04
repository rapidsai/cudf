import itertools
import pickle

import cupy

import cudf
import cudf.core.groupby.groupby

# all (de-)serializtion are attached to cudf Objects:
# Series/DataFrame/Index/Column/Buffer/etc
serializable_classes = (
    cudf.CategoricalDtype,
    cudf.DataFrame,
    cudf.Index,
    cudf.MultiIndex,
    cudf.Series,
    cudf.core.groupby.groupby.GroupBy,
    cudf.core.groupby.groupby._Grouping,
    cudf.core.column.column.ColumnBase,
    cudf.core.buffer.Buffer,
)

try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.utils import log_errors

    @cuda_serialize.register(serializable_classes)
    def cuda_serialize_cudf_object(x):
        with log_errors():
            header, frames = x.serialize()

            assert all((type(f) is cudf.core.buffer.Buffer) for f in frames)
            header["serializer"] = "cuda"
            header["compression"] = (False,) * len(frames)
            header["lengths"] = [f.nbytes for f in frames]

            frames = [
                cupy.concatenate(
                    [cupy.asarray(f) for f in frames]
                ).data.mem._owner
            ]

            return header, frames

    # all (de-)serializtion are attached to cudf Objects:
    # Series/DataFrame/Index/Column/Buffer/etc
    @dask_serialize.register(serializable_classes)
    def dask_serialize_cudf_object(x):
        header, frames = cuda_serialize_cudf_object(x)
        with log_errors():
            header["serializer"] = "dask"
            header["compression"] = (None,) * len(frames)
            frames = [cupy.asnumpy(cupy.asarray(f)).data for f in frames]
            return header, frames

    @cuda_deserialize.register(serializable_classes)
    @dask_deserialize.register(serializable_classes)
    def deserialize_cudf_object(header, frames):
        with log_errors():
            if header["serializer"] == "cuda":
                for f in frames:
                    # some frames are empty -- meta/empty partitions/etc
                    if len(f) > 0:
                        assert hasattr(f, "__cuda_array_interface__")
            elif header["serializer"] == "dask":
                frames = [memoryview(f) for f in frames]

            assert len(frames) == 1
            frames = [
                e.copy().data.mem._owner
                for e in cupy.split(
                    cupy.asarray(frames[0]),
                    itertools.accumulate(header["lengths"][:-1]),
                )
            ]

            cudf_typ = pickle.loads(header["type-serialized"])
            cudf_obj = cudf_typ.deserialize(header, frames)
            return cudf_obj


except ImportError:
    # distributed is probably not installed on the system
    pass
