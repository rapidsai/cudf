try:
    import distributed
except ImportError:
    def register_distributed_serializer(cls):
        pass
else:
    import distributed.protocol as _dp

    def register_distributed_serializer(cls):
        _dp.register_serialization(cls, _serialize, _deserialize)

    def _serialize(df):
        header, frames = df.serialize(_dp.serialize)
        assert 'reconstructor' not in header
        meth_deserial = getattr(type(df), 'deserialize')
        header['reconstructor'] = _dp.serialize(meth_deserial)
        return header, frames

    def _deserialize(header, frames):
        reconstructor = _dp.deserialize(*header['reconstructor'])
        assert reconstructor is not None, 'None {}'.format(header['type'])
        return reconstructor(_dp.deserialize, header, frames)
