import functools


def register_distributed_serializer(cls):
    try:
        from distributed.protocol.cuda import cuda_serialize, cuda_deserialize
        from distributed.protocol import serialize, deserialize

        serialize_part = functools.partial(
            serialize, serializers=["cuda", "dask", "pickle"]
        )
        deserialize_part = functools.partial(
            deserialize, deserializers=["cuda", "dask", "pickle"]
        )

        cuda_serialize.register(cls)(
            functools.partial(cls.serialize, serialize=serialize_part)
        )
        cuda_deserialize.register(cls)(
            functools.partial(cls.deserialize, deserialize_part)
        )
    except ImportError:
        pass
