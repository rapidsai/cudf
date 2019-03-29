#ifndef COPYING_SLICE_HPP
#define COPYING_SLICE_HPP

namespace cudf {

// Forward declaration
struct column_array;

namespace detail {

void slice(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns,
           cudaStream_t*       streams,
           gdf_size_type       streams_size);

}  // namespace detail
}  // namespace cudf

#endif
