from cython.operator cimport dereference as deref
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

from cudf.bindings.cudf_cpp import *
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.utils cimport *
from cudf.dataframe.categorical import CategoricalColumn

cimport cudf.bindings.groupby.hash as hash_groupby


agg_names = {
    'sum': hash_groupby.SUM,
    'min': hash_groupby.MIN,
    'max': hash_groupby.MAX,
    'count': hash_groupby.COUNT,
    'mean': hash_groupby.MEAN
}


def apply_groupby(keys, values, ops, method='hash'):
    """
    Apply aggregations *ops* on *values*, grouping by *keys*.

    Parameters
    ----------
    keys : list of Columns
    values : list of Columns
    ops : str or list of str
        Aggregation to be performed for each column in *values*

    Returns
    -------
    result : tuple of list of Columns
        keys and values of the result
    """
    if len(values) == 0:
        return (keys, [])

    cdef pair[cudf_table, cudf_table] result
    cdef cudf_table *c_keys_table = table_from_columns(keys)
    cdef cudf_table *c_values_table = table_from_columns(values)
    cdef vector[hash_groupby.operators] c_ops

    num_values_cols = len(values)
    for i in range(num_values_cols):
        if isinstance(ops, str):
            c_ops.push_back(agg_names[ops])
        else:
            c_ops.push_back(agg_names[ops[i]])

    cdef hash_groupby.Options *options = new hash_groupby.Options()

    result = hash_groupby.groupby(
        c_keys_table[0],
        c_values_table[0],
        c_ops,
        deref(options)
    )

    result_key_cols = columns_from_table(&result.first)
    result_value_cols = columns_from_table(&result.second)

    for i, inp_key_col in enumerate(keys):
        if isinstance(inp_key_col, CategoricalColumn):
            result_key_cols[i] = CategoricalColumn(
                data=result_key_cols[i].data,
                mask=result_key_cols[i].mask,
                categories=inp_key_col.cat().categories,
                ordered=inp_key_col.cat().ordered
            )

    for i, inp_value_col in enumerate(values):
         if isinstance(inp_value_col, CategoricalColumn):
             result_value_cols[i] = CategoricalColumn(
                 data=result_value_cols[i].data,
                 mask=result_value_cols[i].mask,
                 categories=inp_value_col.cat().categories,
                 ordered=inp_value_col.cat().ordered
             )

    return (result_key_cols, result_value_cols)
