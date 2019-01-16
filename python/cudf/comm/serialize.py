# A flag to allow dask_gdf to detect and warn if
# IPC serialization is unavailable
CUSTOM_SERIALIZATION_AVAILABLE = False


def register_distributed_serializer(cls):
    """ Don't do custom serialization for cudf """
    pass
