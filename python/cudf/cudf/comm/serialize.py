import pickle

import cudf
import cudf.core.groupby.groupby

try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.utils import log_errors

    # all (de-)serializtion are attached to cudf Objects:
    # Series/DataFrame/Index/Column/Buffer/etc
    @cuda_serialize.register(
        (
            cudf.DataFrame,
            cudf.Series,
            cudf.core.series.Series,
            cudf.core.groupby.groupby._Groupby,
            cudf.core.column.column.Column,
            cudf.core.buffer.Buffer,
        )
    )
    def cuda_serialize_cudf_dataframe(x):
        with log_errors():
            header, frames = x.serialize()
            return header, frames

    # all (de-)serializtion are attached to cudf Objects:
    # Series/DataFrame/Index/Column/Buffer/etc
    @dask_serialize.register(
        (
            cudf.DataFrame,
            cudf.Series,
            cudf.core.series.Series,
            cudf.core.groupby.groupby._Groupby,
            cudf.core.column.column.Column,
            cudf.core.buffer.Buffer,
        )
    )
    def dask_serialize_cudf_dataframe(x):
        with log_errors():
            header, frames = x.serialize()
            assert all(isinstance(f, cudf.core.buffer.Buffer) for f in frames)
            frames = [f.to_host_array().data for f in frames]
            return header, frames

    @cuda_deserialize.register(
        (
            cudf.DataFrame,
            cudf.Series,
            cudf.core.series.Series,
            cudf.core.groupby.groupby._Groupby,
            cudf.core.column.column.Column,
            cudf.core.buffer.Buffer,
        )
    )
    @dask_deserialize.register(
        (
            cudf.DataFrame,
            cudf.Series,
            cudf.core.series.Series,
            cudf.core.groupby.groupby._Groupby,
            cudf.core.column.column.Column,
            cudf.core.buffer.Buffer,
        )
    )
    def deserialize_cudf_dataframe(header, frames):
        with log_errors():
            cudf_typ = pickle.loads(header["type-serialized"])
            cudf_obj = cudf_typ.deserialize(header, frames)
            return cudf_obj


except ImportError:
    # distributed is probably not installed on the system
    pass
