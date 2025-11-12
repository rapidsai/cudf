/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

namespace detail {

enum class negative_index_policy : bool { ALLOWED, NOT_ALLOWED };

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
 * @param[in] bounds_policy How to treat out-of-bounds indices. `NULLIFY` coerces rows that
 * correspond to out-of-bounds indices in the gather map to be null elements. For better
 * performance, use `DONT_CHECK` when the `gather_map` is known to contain only valid
 * indices. If `policy` is set to `DONT_CHECK` and there are out-of-bounds indices in `gather_map`,
 * the behavior is undefined.
 * @param[in] negative_index_policy Interpret each negative index `i` in the
 * `gather_map` as the positive index `i+num_source_rows`.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return Result of the gather
 */
std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::detail::gather(table_view const&,column_view const&,table_view
 * const&,cudf::out_of_bounds_policy,cudf::detail::negative_index_policy,rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 * @throws cudf::logic_error if `gather_map` span size is larger than max of `size_type`.
 */
std::unique_ptr<table> gather(table_view const& source_table,
                              device_span<size_type const> const gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
