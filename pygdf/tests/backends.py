"""Provides access to APIs of different CUDA backends needed for
testing.
"""
import pytest

# List of cuda backends in the form of API function mapping:
cuda_api = []


def cuda_backend_test(fn):
    return pytest.mark.parametrize("cuda", cuda_api,
                                   ids=[api._name for api in cuda_api])(fn)


class AttrDict(dict):

    __getattr__ = dict.__getitem__

    def __repr__(self):
        return self['_name']


# Backend: numba.cuda
try:
    from numba import cuda as nb_cuda
except ImportError as msg:
    print('Failed to import pyarrow.cuda: {}'.format(msg))
    nb_cuda = None

if nb_cuda is not None and 0:
    from pygdf.gpuarrow import GpuArrowReader
    cuda_api.append(
        AttrDict(
            _name='numba.cuda',
            to_device=nb_cuda.to_device,
            context=nb_cuda.current_context,
            ArrowReader=GpuArrowReader,
        )
    )


# Backend: CUDA enabled pyarrow
try:
    import pyarrow.cuda as pa_cuda
except ImportError as msg:
    print('Failed to import pyarrow.cuda: {}'.format(msg))
    pa_cuda = None

if pa_cuda is not None:
    from pygdf.cudaarrow import CudaArrowReader
    from pygdf.cudaarray import CudaNDArray
    cuda_api.append(
        AttrDict(
            _name='pyarrow.cuda',
            to_device=CudaNDArray.fromarray,
            ArrowReader=CudaArrowReader,
            context=pa_cuda.Context
        ))

# Backend: rmm?
