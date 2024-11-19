# Copyright (c) 2020-2024, NVIDIA CORPORATION.


from libcpp.string cimport string

from libcpp.vector cimport vector

from pylibcudf.libcudf.io.types cimport column_name_info

from cudf._lib.column cimport Column

from cudf.core.dtypes import StructDtype

cdef add_df_col_struct_names(df, child_names_dict):
    for name, child_names in child_names_dict.items():
        col = df._data[name]

        df._data[name] = update_col_struct_field_names(col, child_names)


cdef update_col_struct_field_names(Column col, child_names):
    if col.children:
        children = list(col.children)
        for i, (child, names) in enumerate(zip(children, child_names.values())):
            children[i] = update_col_struct_field_names(
                child,
                names
            )
        col.set_base_children(tuple(children))

    if isinstance(col.dtype, StructDtype):
        col = col._rename_fields(
            child_names.keys()
        )

    return col


cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info
):
    # Deprecated, remove in favor of add_col_struct_names
    # when a reader is ported to pylibcudf
    for i, (name, col) in enumerate(table._column_labels_and_values):
        table._data[name] = update_column_struct_field_names(
            col, schema_info[i]
        )


cdef Column update_column_struct_field_names(
    Column col,
    column_name_info& info
):
    cdef vector[string] field_names

    if col.children:
        children = list(col.children)
        for i, child in enumerate(children):
            children[i] = update_column_struct_field_names(
                child,
                info.children[i]
            )
        col.set_base_children(tuple(children))

    if isinstance(col.dtype, StructDtype):
        field_names.reserve(len(col.base_children))
        for i in range(info.children.size()):
            field_names.push_back(info.children[i].name)
        col = col._rename_fields(
            field_names
        )

    return col
