from numba.core.types.abstract import Type
import numpy as np
import pandas as pd

import cudf
from cudf.core.column import as_column
from cudf.utils.dtypes import (
    can_convert_to_column, is_list_like,
    is_numerical_dtype,
    is_datetime_dtype,
    is_timedelta_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_list_dtype,
    is_struct_dtype,
)

def to_numeric(arg, errors='raise', downcast=None):
    """
    Convert argument into numerical types.

    Parameters
    ----------
    arg : column-convertible
        The object to convert to numeric types
    errors : {'raise', 'ignore', 'coerce'}, defaults 'raise'
        Policy to handle errors during parsing. 
        - 'raise' will notify user all errors encountered. 
        - 'ignore' will skip error and returns input. 
        - 'coerce' will leave invalid values as nulls.
    downcast : optional {'integer', 'signed', 'unsigned', 'float'}, defaults None
        If set, will try to down-convert the datatype of the
        parsed results to smallest possible type. For each `downcast`
        type, this method will determine the smallest possible
        dtype from the following sets:
        - {'integer', 'signed'}: np.typecodes['Integer']
        - {'unsigned'}: np.typecodes['UnsignedInteger']
        - {'float'}: np.typecodes['Float'], excluding types smaller than `float32`
        Note that errors encountered in downcasts will be raised regardless of
        `error` parameter.

    Returns
    -------
    Depending on the input, if series is passed in, series is returned,
    otherwise ndarray

    Examples
    --------
    """

    if errors not in {'raise', 'ignore', 'coerce'}:
        raise ValueError('invalid error value specified')

    if downcast not in {None, 'integer', 'signed', 'unsigned', 'float'}:
        raise ValueError('invalid downcasting method provided')

    if not can_convert_to_column(arg) or (hasattr(arg, 'ndim') and arg.ndim > 1):
        raise ValueError('arg must be column convertible')
    
    col = as_column(arg)
    dtype = col.dtype

    if is_datetime_dtype(dtype) or is_timedelta_dtype(dtype):
        col = col.as_numerical_column(np.dtype('int64'))
    elif is_categorical_dtype(dtype):
        cat_dtype = col.dtype.type
        if is_numerical_dtype(cat_dtype):
            col = col.as_numerical_column(cat_dtype)
        else:
            try:
                col = _convert_str_col(col._get_decategorized_column(), errors)
            except ValueError as e:
                if errors == 'ignore':
                    return arg
                else:
                    raise e
    elif is_string_dtype(dtype):
        try:
            col = _convert_str_col(col, errors)
        except ValueError as e:
            if errors == 'ignore':
                return arg
            else:
                raise e
    elif is_list_dtype(dtype) or is_struct_dtype(dtype):
        raise ValueError('Input does not support nested datatypes')
    elif is_numerical_dtype(dtype):
        pass
    else:
        raise ValueError('Unrecognized datatype')

    if downcast:
        downcast_type_map = {
            'integer': list(np.typecodes['Integer']),
            'signed': list(np.typecodes['Integer']),
            'unsigned': list(np.typecodes['UnsignedInteger']),
        }
        float_types = list(np.typecodes['Float'])
        idx = float_types.index(np.dtype(np.float32).char)
        downcast_type_map['float'] = float_types[idx:]

        type_set = downcast_type_map[downcast]
        downcast_dtype = next(np.dtype(t) for t in type_set if np.dtype(t).itemsize < col.dtype.itemsize)

        col = col.astype(downcast_dtype)
    
    if isinstance(arg, (cudf.Series, pd.Series)):
        return cudf.Series(col)
    else:
        return col.to_array(fillna='pandas')

def _convert_str_col(col, errors):
    if not is_string_dtype(col):
        raise TypeError("col must be string dtype.")
    
    is_integer = col.str().isinteger()
    if is_integer.all():
        return col.as_numerical_column(dtype=np.dtype("i8"))
    is_float = col.str().isfloat()
    if is_float.all():
        return col.as_numerical_column(dtype=np.dtype('d'))
    if is_integer.sum() + is_float.sum() == len(col):
        return col.as_numerical_column(dtype=np.dtype('d'))
    # TODO: account for inf strings "[+|-]?(inf|infinity)"
    else:
        if errors == 'coerce':
            return col.as_numerical_column(dtype=np.dtype('d'))
        else:
            raise ValueError('Unable to convert some strings to numerics.')
