#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/table/table.hpp>

#include <memory>


namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the source table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param[in] check_bounds Optionally perform bounds checking on the values
 * of `gather_map` and throw an error if any of its values are out of bounds.
 * @param[in] ignore_out_of_bounds Ignore values in `gather_map` that are
 * out of bounds. Currently incompatible with `allow_negative_indices`,
 * i.e., setting both to `true` is undefined.
 * @param[in] allow_negative_indices Interpret each negative index `i` in the
 * gathermap as the positive index `i+num_source_rows`.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The CUDA stream on which to execute kernels
 * @return cudf::table Result of the gather
 */
std::unique_ptr<table> gather(table_view const& source_table, column_view const& gather_map,
			      bool check_bounds = false, bool ignore_out_of_bounds = false,
			      bool allow_negative_indices = false,
			      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
			      cudaStream_t stream = 0);
}  // namespace detail
}  // namespace exp
}  // namespace cudf
