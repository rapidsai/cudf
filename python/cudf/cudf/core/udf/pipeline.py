from numba.np import numpy_support
from nvtx import annotate

from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def compile_masked_udf(func, dtypes):
    """
    Generate an inlineable PTX function that will be injected into
    a variadic kernel inside libcudf

    assume all input types are `MaskedType(input_col.dtype)` and then
    compile the requestied PTX function as a function over those types
    """
    to_compiler_sig = tuple(
        MaskedType(arg)
        for arg in (numpy_support.from_dtype(np_type) for np_type in dtypes)
    )
    # Get the inlineable PTX function
    ptx, numba_output_type = cudautils.compile_udf(func, to_compiler_sig)
    numpy_output_type = numpy_support.as_dtype(numba_output_type.value_type)

    return numpy_output_type, ptx


def nulludf(func):
    """
    Mimic pandas API:

    def f(x, y):
        return x + y
    df.apply(lambda row: f(row['x'], row['y']))

    in this scheme, `row` is actually the whole dataframe
    `DataFrame` sends `self` in as `row` and subsequently
    we end up calling `f` on the resulting columns since
    the dataframe is dict-like
    """

    def wrapper(*args):
        from cudf import DataFrame

        # This probably creates copies but is fine for now
        to_udf_table = DataFrame(
            {idx: arg for idx, arg in zip(range(len(args)), args)}
        )
        # Frame._apply
        return to_udf_table._apply(func)

    return wrapper
