# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.strings.side_type cimport side_type
from cudf._lib.pylibcudf.scalar cimport Scalar

cpdef Column strip(
    Column input,
    side_type side,
    Scalar to_strip
):
    """Removes the specified characters from the beginning 
    or end (or both) of each string.

    For details, see :cpp:func:`cudf::strings::strip`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    side : SideType, default SideType.BOTH
    	Indicates characters are to be stripped from the beginning, 
        end, or both of each string; Default is both
    to_strip : Scalar
        UTF-8 encoded characters to strip from each string;
        Default is empty string which indicates strip whitespace characters

    Returns
    -------
    pylibcudf.Column
        New strings column.
    """

    pass
    
