namespace cudf {

namespace detail {

    void shift(
        table *out_table,
        table const &in_table,
        gdf_scalar const* fill_value,
        gdf_index_type copy_begin,
        gdf_index_type dest_begin,
        gdf_index_type dest_end,
        gdf_index_type fill_begin,
        gdf_index_type fill_end
    )
    {
        for (gdf_index_type i = 0; i < out_table->num_columns(); i++)
        {
            auto out_column = out_table->get_column(i);
            auto in_column = in_table.get_column(i);
    
            cudf::copy_range(out_column, *in_column, dest_begin, dest_end, copy_begin);
            cudf::fill(out_column, *fill_value, fill_begin, fill_end);
        }
    }
    
}; // namespace: detail

}; // namespace: cudf
