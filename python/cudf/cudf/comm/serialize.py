import pickle

import cudf
import cudf.core.groupby.groupby

try:
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
        )
    )
    def serialize_cudf_dataframe(x):
        with log_errors():
            header, frames = x.serialize()
            return header, frames

    @cuda_deserialize.register(
        (
            cudf.DataFrame,
            cudf.Series,
            cudf.core.series.Series,
            cudf.core.groupby.groupby._Groupby,
            cudf.core.column.column.Column,
        )
    )
    def deserialize_cudf_dataframe(header, frames):
        with log_errors():
            cudf_typ = pickle.loads(header["type"])
            cudf_obj = cudf_typ.deserialize(header, frames)
            return cudf_obj


except ImportError:
    # distributed is probably not installed on the system
    pass
