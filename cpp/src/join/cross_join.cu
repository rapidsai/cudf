#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <hash/concurrent_unordered_map.cuh>

#include <join/join_common_utils.hpp>

#include <cudf/detail/gather.cuh>
#include <join/hash_join.cuh>
#include "cudf/column/column.hpp"
#include "cudf/filling.hpp"
#include "cudf/scalar/scalar_factories.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/type_dispatcher.hpp"

namespace cudf {
namespace detail {
/**
 * @brief  Performs a cross join on the specified columns of two
 * tables (left, right)
 *
 * The cross join returns the cartesian product of rows from each table.
 *
 * The approach is to repeat the left table by the number of rows in the right table
 * and tile the right table by the number of rows in the left table.
 *
 * @throws cudf::logic_error if number of columns in either `left` or `right` table is 0
 * @throws cudf::logic_error if number of returned columns is 0
 *
 * @param[in] left                  The left table
 * @param[in] right                 The right table
 * @param[in] return_left_columns   A vector of column indices from `left` to include in the
 *                                  returned table.
 * @param[in] return_right_columns  A vector of column indices from `right` to include in the
 *                                  returned table.
 * @param[in] mr                    Device memory resource to use for device memory allocation
 * @param[in] stream                Cuda stream
 *
 * @returns                         Result of cross joining `left` and `right` tables, keeping
 *                                  columns specified by `return_left_columns` and
 *                                  `return_right_columns`.
 */
std::unique_ptr<cudf::table> cross_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& return_left_columns,
  std::vector<cudf::size_type> const& return_right_columns,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  CUDF_EXPECTS(0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS(0 != right.num_columns(), "Right table is empty");

  // Only repeat/tile the selected columns
  auto left_selected{left.select(return_left_columns)};
  auto right_selected{right.select(return_right_columns)};

  // If left or right table has no rows, return an empty table with all selected columns
  if ((0 == left.num_rows()) || (0 == right.num_rows())) {
    auto empty_left{empty_like(left_selected)};
    auto empty_right{empty_like(right_selected)};
    // TODO: Return empty table with all selected columns
  }

  // Repeat left table
  numeric_scalar<size_type> num_repeats{right.num_rows()};
  auto left_repeated{cudf::repeat(left_selected, num_repeats, mr)};

  // Tile right table
  auto right_tiled{cudf::tile(right_selected, left.num_rows(), mr)};

  // Concatenate all repeated/tiled columns into one table
  auto left_repeated_columns{left_repeated->release()};
  auto right_tiled_columns{right_tiled->release()};
  std::move(right_tiled_columns.begin(),
            right_tiled_columns.end(),
            std::back_inserter(left_repeated_columns));

  return std::make_unique<table>(std::move(left_repeated_columns));
}
}  // namespace detail

std::unique_ptr<cudf::table> cross_join(cudf::table_view const& left,
                                        cudf::table_view const& right,
                                        std::vector<cudf::size_type> const& return_left_columns,
                                        std::vector<cudf::size_type> const& return_right_columns,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::cross_join(left, right, return_left_columns, return_right_columns, mr, 0);
}

}  // namespace cudf
