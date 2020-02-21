# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf._lib.GDFError import GDFError
from cudf._libxx import dlpack as libdlpack
from cudf.core.column import ColumnBase
from cudf.core.dataframe import DataFrame
from cudf.core.index import Index
from cudf.core.series import Series
from cudf.utils import ioutils


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
        res = libdlpack.from_dlpack(pycapsule_obj)
    except GDFError as err:
        if str(err) == "b'GDF_DATASET_EMPTY'":
            raise ValueError(
                "Cannot create a cuDF Object from a DLPack tensor of 0 size"
            )
        else:
            raise err

    if len(res._data) == 1:
        return Series(res._data[0])
    else:
        return DataFrame(data=res._data)


@ioutils.doc_to_dlpack()
def to_dlpack(cudf_obj):
    """{docstring}"""
    if len(cudf_obj) == 0:
        raise ValueError("Cannot create DLPack tensor of 0 size")

    if isinstance(cudf_obj, (DataFrame, Series, Index)):
        gdf_cols = cudf_obj
    elif isinstance(cudf_obj, ColumnBase):
        gdf_cols = cudf_obj.as_frame()
    else:
        raise TypeError(
            f"Input of type {type(cudf_obj)} cannot be converted "
            "to DLPack tensor"
        )

    return libdlpack.to_dlpack(gdf_cols)
