import functools

def register_distributed_serializer(cls):
    try: 
        from distributed.protocol.cuda import cuda_serialize, cuda_deserialize
        from distributed.protocol import serialize, deserialize
        
        cuda_serialize.register(cls)(functools.partial(cls.serialize, serialize=serialize))
        cuda_deserialize.register(cls)(functools.partial(cls.deserialize, deserialize))
    except ImportError:
        pass
