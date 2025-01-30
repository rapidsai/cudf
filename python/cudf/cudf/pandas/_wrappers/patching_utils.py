import cudf
import functools

# Save the original __init__ methods
_original_Series_init = cudf.Series.__init__
_original_DataFrame_init = cudf.DataFrame.__init__
_original_Index_init = cudf.Index.__init__

def wrap_init(original_init):
    @functools.wraps(original_init)
    def wrapped_init(self, data=None, *args, **kwargs):
        print("wrapped_init")
        if is_proxy_object(data):
            print("data is a proxy object")
            data = data.as_gpu_object()
        original_init(self, data, *args, **kwargs)
    return wrapped_init


@functools.wraps(_original_DataFrame_init)
def DataFrame_init_(self, data, index=None, columns=None, *args, **kwargs):
    if is_proxy_object(data):
        data = data.as_gpu_object()
    if is_proxy_object(index):
        index = index.as_gpu_object()
    if is_proxy_object(columns):
        columns = columns.as_cpu_object()
    _original_DataFrame_init(self, data, index, columns, *args, **kwargs)


def patch_inits_methods():
    # Replace the __init__ methods with the wrapped versions
    cudf.Series.__init__ = wrap_init(_original_Series_init)
    cudf.Index.__init__ = wrap_init(_original_Index_init)
    cudf.DataFrame.__init__ = DataFrame_init_


def undo_inits_patching():
    cudf.Series.__init__ = _original_Series_init
    cudf.DataFrame.__init__ = _original_DataFrame_init
    cudf.Index.__init__ = _original_Index_init
