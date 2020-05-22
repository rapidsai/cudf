#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/table/table.hpp>

#include <memory>

namespace cudf {
namespace detail {

enum class out_of_bounds_policy : int8_t { FAIL, NULLIFY, IGNORE };

enum class negative_indices_policy : bool { ALLOW, NOT_ALLOWED };

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
 * @throws cudf::logic_error if `check_bounds == true` and an index exists in
 * `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the source table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param[in] out_of_bounds_policy Specifies how to treat out of bounds indices. FAIL means checking
 * if the values of `gather_map` and throwing an error if any of its values are out of bounds.
 * NULLIFY means to nullify indices that are outside the bounds. IGNORE means to ignore values in
 * `gather_map` that are out of bounds. The IGNORE option is currently incompatible when
 * `negative_indices_policy` is set to ALLOW.
 * @param[in] negative_indices_policy Interpret each negative index `i` in the
 * gathermap as the positive index `i+num_source_rows`.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The CUDA stream on which to execute kernels
 * @return cudf::table Result of the gather
 */
std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds,
                              negative_indices_policy neg_indices,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream                 = 0);
}  // namespace detail
}  // namespace cudf
