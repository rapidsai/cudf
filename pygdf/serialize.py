import os
import sys
from types import MethodType

# A flag to allow dask_gdf to detect and warn if
# IPC serialization is unavailable
CUSTOM_SERIALIZATION_AVAILABLE = False

try:
    import distributed.protocol as _dp
    from distributed.utils import has_keyword
except ImportError:
    def register_distributed_serializer(cls):
        """Dummy no-op function.
        """
        pass
else:
    CUSTOM_SERIALIZATION_AVAILABLE = True

    def register_distributed_serializer(cls):
        """Register serialization methods for dask.distributed.
        """
        _dp.register_serialization(cls, _serialize, _deserialize)

    def has_context_keyword(meth):
        if isinstance(meth, MethodType):
            return has_keyword(meth.__func__, 'context')
        else:
            return has_keyword(meth, 'context')

    def _serialize(df, context=None):
        def do_serialize(x):
            return _dp.serialize(x, context=context)

        def call_with_context(meth, x):
            if has_context_keyword(meth):
                return meth(x, context=context)
            else:
                return meth(x)

        header, frames = call_with_context(df.serialize, do_serialize)
        assert 'reconstructor' not in header
        meth_deserial = getattr(type(df), 'deserialize')
        header['reconstructor'] = do_serialize(meth_deserial)
        return header, frames

    def _deserialize(header, frames):
        reconstructor = _dp.deserialize(*header['reconstructor'])
        assert reconstructor is not None, 'None {}'.format(header['type'])
        return reconstructor(_dp.deserialize, header, frames)


def _parse_transfer_context(context):
    from distributed.comm.addressing import parse_host_port, parse_address

    def parse_it(x):
        return parse_host_port(parse_address(x)[1])

    if 'recipient' in context and 'sender' in context:
        rechost, recport = parse_it(context['recipient'])
        senhost, senport = parse_it(context['sender'])
        same_node = rechost == senhost
        same_process = same_node and recport == senport
    else:
        same_node, same_process = False, False
    return same_node, same_process


def should_use_ipc(context):
    """Use destination context info to determine if we should use CUDA-IPC.

    Parameters
    ----------
    context : dict or None
        If not ``None``, it contains information about the destination.
        See custom serialization in dask.distributed.

    Returns
    -------
    return_value : bool
        ``True`` if it is possible to perform CUDA IPC transfer to the
        destination.
    """
    _CONFIG_USE_IPC = bool(int(os.environ.get("DASK_GDF_USE_IPC", "1")))

    # User opt-out
    if not _CONFIG_USE_IPC:
        return False
    # CUDA IPC is only supported on Linux
    if not sys.platform.startswith('linux'):
        return False
    # *context* is not given.
    if context is None:
        return False
    # Check if destination on the same
    same_node, same_process = _parse_transfer_context(context)
    return bool(same_node)
