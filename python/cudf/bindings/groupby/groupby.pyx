from cython.operator cimport dereference as deref
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cudf.bindings.cudf_cpp import *
from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.types cimport table as cudf_table
from cudf.bindings.utils cimport table_from_dataframe, dataframe_from_table

cimport cudf.bindings.groupby.hash as hash_groupby


agg_names = {
    'sum': hash_groupby.SUM,
    'min': hash_groupby.MIN,
    'max': hash_groupby.MAX,
    'count': hash_groupby.COUNT
}


def apply_groupby(values, keys, ops, method='hash'):
    """
    Apply aggregations *ops* on *values*, grouping by *keys*.

    Parameters
    ----------
    keys : DataFrame
    values : DataFrame
    ops : str or list of str
        Aggregation to be performed for each column in *values*

    Returns
    -------
    result : tuple of DataFrame
        DataFrames containing keys and values of the result
    """
    cdef pair[cudf_table, cudf_table] result
    cdef cudf_table *c_keys_table = table_from_dataframe(keys)
    cdef cudf_table *c_values_table = table_from_dataframe(values)
    cdef vector[hash_groupby.operators] c_ops

    num_values_cols = values.shape[1]
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

    return (
        dataframe_from_table(&result.first, keys.columns),
        dataframe_from_table(&result.second, values.columns)
    )
