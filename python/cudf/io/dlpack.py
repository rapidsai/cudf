# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.dataframe.index import Index
from cudf.dataframe.column import Column
from cudf.dataframe.buffer import Buffer
from cudf.dataframe import columnops
from cudf.utils import ioutils
from cudf.bindings import dlpack as cpp_dlpack
from cudf.bindings.GDFError import GDFError


def from_dlpack(pycapsule_obj):
    """Converts from a DLPack tensor to a cuDF object.

    DLPack is an open-source memory tensor structure:
    `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

    This function takes a PyCapsule object which contains a pointer to
    a DLPack tensor as input, and returns a cuDF object. This function deep
    copies the data in the DLPack tensor into a cuDF object.

    Parameters
    ----------
    pycapsule_obj : PyCapsule
        Input DLPack tensor pointer which is encapsulated in a PyCapsule
        object.

    Returns
    -------
    A cuDF DataFrame or Series depending on if the input DLPack tensor is 1D
    or 2D.
    """
    try:
        res, valids = cpp_dlpack.from_dlpack(pycapsule_obj)
    except GDFError as err:
        if str(err) == "b'GDF_DATASET_EMPTY'":
            raise ValueError(
                "Cannot create a cuDF Object from a DLPack tensor of 0 size"
            )
        else:
            raise err
    cols = []
    for idx in range(len(valids)):
        mask = None
        if valids[idx]:
            mask = Buffer(valids[idx])
        cols.append(
            columnops.build_column(
                Buffer(res[idx]),
                dtype=res[idx].dtype,
                mask=mask
            )
        )
    if len(cols) == 1:
        return Series(cols[0])
    else:
        df = DataFrame()
        for idx, col in enumerate(cols):
            df[idx] = col
        return df


@ioutils.doc_to_dlpack()
def to_dlpack(cudf_obj):
    """{docstring}"""
    if len(cudf_obj) == 0:
        raise ValueError("Cannot create DLPack tensor of 0 size")

    if isinstance(cudf_obj, DataFrame):
        gdf_cols = [col[1]._column for col in cudf_obj._cols.items()]
    elif isinstance(cudf_obj, Series):
        gdf_cols = [cudf_obj._column]
    elif isinstance(cudf_obj, Index):
        gdf_cols = [cudf_obj._values]
    elif isinstance(cudf_obj, Column):
        gdf_cols = [cudf_obj]
    else:
        raise TypeError(f"Input of type {type(cudf_obj)} cannot be converted "
                        "to DLPack tensor")

    return cpp_dlpack.to_dlpack(gdf_cols)
