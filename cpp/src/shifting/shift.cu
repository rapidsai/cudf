#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>

#include <vector>

#include "shift.cuh"

namespace cudf {

void shift(table *out_table, table const &in_table, gdf_index_type offset, gdf_scalar const* fill_value)
{
    // TODO(cwharris): assert in / out same row count
    auto num_rows = out_table->num_rows();

    if (offset > 0) {
        auto mid = offset;
        detail::shift(
            out_table,
            in_table,
            fill_value,
            0,          // copy_begin
            mid,        // dest_begin
            num_rows,   // dest_end
            0,          // fill_begin
            mid         // fill_end
        );
    } else {
        auto mid = num_rows + offset;
        detail::shift(
            out_table,
            in_table,
            fill_value,
            -offset,    // copy_begin
            0,          // dest_begin
            mid,        // dest_end
            mid,        // fill_begin
            num_rows    // fill_end
        );
    }
}

}; // namespace cudf