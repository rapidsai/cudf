#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <cudf/copying.hpp>

namespace cudf {

void shift(table *out_table, table const &in_table, gdf_index_type offset)
{
    for (gdf_index_type i = 0; i < out_table->num_columns(); i++)
    {
        auto out_column = out_table->get_column(i);
        auto in_column = in_table.get_column(i);

        if (offset > 0) {
            cudf::copy_range(out_column, *in_column, offset, out_column->size, 0);
        } else {

        }
    }
}

}; // namespace cudf