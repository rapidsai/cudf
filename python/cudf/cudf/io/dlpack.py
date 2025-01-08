# Copyright (c) 2019-2024, NVIDIA CORPORATION.
from __future__ import annotations

import pylibcudf as plc

import cudf
from cudf.core.column import ColumnBase
from cudf.utils import ioutils


def from_dlpack(pycapsule_obj) -> cudf.Series | cudf.DataFrame:
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

    Notes
    -----
    cuDF from_dlpack() assumes column-major (Fortran order) input. If the input
    tensor is row-major, transpose it before passing it to this function.
    """
    plc_table = plc.interop.from_dlpack(pycapsule_obj)
    data = dict(
        enumerate(
            (ColumnBase.from_pylibcudf(col) for col in plc_table.columns())
        )
    )

    if len(data) == 1:
        return cudf.Series._from_data(data)
    else:
        return cudf.DataFrame._from_data(data)


@ioutils.doc_to_dlpack()
def to_dlpack(cudf_obj: cudf.Series | cudf.DataFrame | cudf.BaseIndex):
    """Converts a cuDF object to a DLPack tensor.

    DLPack is an open-source memory tensor structure:
    `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

    This function takes a cuDF object as input, and returns a PyCapsule object
    which contains a pointer to DLPack tensor. This function deep copies
    the data in the cuDF object into the DLPack tensor.

    Parameters
    ----------
    cudf_obj : cuDF Object
        Input cuDF object.

    Returns
    -------
    A  DLPack tensor pointer which is encapsulated in a PyCapsule object.

    Notes
    -----
    cuDF to_dlpack() produces column-major (Fortran order) output. If the
    output tensor needs to be row major, transpose the output of this function.
    """
    if isinstance(cudf_obj, (cudf.DataFrame, cudf.Series, cudf.BaseIndex)):
        gdf = cudf_obj
    elif isinstance(cudf_obj, ColumnBase):
        gdf = cudf.Series._from_column(cudf_obj)
    else:
        raise TypeError(
            f"Input of type {type(cudf_obj)} cannot be converted "
            "to DLPack tensor"
        )

    if any(
        not cudf.api.types._is_non_decimal_numeric_dtype(dtype)
        for _, dtype in gdf._dtypes  # type: ignore[union-attr]
    ):
        raise TypeError("non-numeric data not yet supported")

    dtype = cudf.utils.dtypes.find_common_type(
        [dtype for _, dtype in gdf._dtypes]  # type: ignore[union-attr]
    )
    gdf = gdf.astype(dtype)
    return plc.interop.to_dlpack(
        plc.Table([col.to_pylibcudf(mode="read") for col in gdf._columns])
    )
