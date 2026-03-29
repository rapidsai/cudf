/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_merge_join_impl.cuh"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>

#include <cub/device/device_transform.cuh>
#include <thrust/scatter.h>

namespace cudf::detail {

namespace {

struct is_row_null {
  bitmask_type const* const _validity_mask;

  __device__ auto operator()(size_type idx) const noexcept
  {
    return !cudf::bit_is_set(_validity_mask, idx);
  }
};

}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           sorted is_left_sorted,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::left_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.left(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);

      //  For left join with UNEQUAL nulls, we need to add back rows that were filtered out.
      //  Remaining configs can return directly
      if (compare_nulls == null_equality::EQUAL ||
          !has_nested_nulls(preprocessed_left._table_view)) {
        return std::pair{std::move(preprocessed_left_indices),
                         std::move(preprocessed_right_indices)};
      }

      // Special handling for UNEQUAL null semantics with nested nulls:
      // Rows containing nulls were filtered during preprocessing and must be reinserted.
      // These rows have no matches by definition (nulls are unequal), so they're added
      // to the output with JoinNoMatch sentinel values for the right side.

      auto const num_filtered_nulls = preprocessed_left._num_nulls.value();
      auto const total_output_size =
        preprocessed_left_indices->size() + static_cast<int64_t>(num_filtered_nulls);

      // Create new result vectors with space for filtered rows
      rmm::device_uvector<size_type> left_result_indices(total_output_size, stream, mr);
      rmm::device_uvector<size_type> right_result_indices(total_output_size, stream, mr);

      // Copy existing join results
      {
        using Iterator       = decltype(preprocessed_left_indices->begin());
        auto input_iterators = cudf::detail::make_pinned_vector_async<Iterator>(2, stream);
        input_iterators[0]   = preprocessed_left_indices->begin();
        input_iterators[1]   = preprocessed_right_indices->begin();

        auto output_iterators = cudf::detail::make_pinned_vector_async<Iterator>(2, stream);
        output_iterators[0]   = left_result_indices.begin();
        output_iterators[1]   = right_result_indices.begin();

        auto sizes = cudf::detail::make_pinned_vector_async<size_t>(2, stream);
        sizes[0]   = preprocessed_left_indices->size();
        sizes[1]   = preprocessed_right_indices->size();

        sort_merge_join_detail::batched_copy(
          input_iterators.begin(), output_iterators.begin(), sizes.begin(), 2, stream);
      }

      // Append filtered null rows with JoinNoMatch for right side
      auto const validity_mask =
        static_cast<bitmask_type const*>(preprocessed_left._validity_mask.value().data());
      cudf::detail::copy_if_async(cuda::counting_iterator<size_type>(0),
                                  cuda::counting_iterator<size_type>(left.num_rows()),
                                  cuda::counting_iterator<size_type>(0),
                                  left_result_indices.begin() + preprocessed_left_indices->size(),
                                  is_row_null{validity_mask},
                                  stream);
      cub::DeviceTransform::Fill(right_result_indices.begin() + preprocessed_right_indices->size(),
                                 num_filtered_nulls,
                                 JoinNoMatch,
                                 stream.value());

      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(std::move(left_result_indices)),
        std::make_unique<rmm::device_uvector<size_type>>(std::move(right_result_indices))};
    },
    stream);
}

std::unique_ptr<cudf::join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left,
  sorted is_left_sorted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join_match_context"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, left, &preprocessed_left, stream, mr](auto& obj) mutable {
      auto matches_per_row = obj.matches_per_row(stream, cudf::get_current_device_resource_ref());
      matches_per_row->resize(matches_per_row->size() - 1, stream);
      if (compare_nulls == null_equality::UNEQUAL &&
          has_nested_nulls(preprocessed_left._table_view)) {
        // Now we need to post-process the matches i.e. insert zero counts for all the null
        // positions
        auto unprocessed_matches_per_row =
          cudf::detail::make_zeroed_device_uvector_async<size_type>(
            preprocessed_left._table_view.num_rows(), stream, mr);
        auto mapping = preprocessed_left.map_table_to_unprocessed(stream);
        thrust::scatter(rmm::exec_policy_nosync(stream),
                        matches_per_row->begin(),
                        matches_per_row->end(),
                        mapping.begin(),
                        unprocessed_matches_per_row.begin());
        return std::make_unique<sort_merge_join_match_context>(
          left,
          std::make_unique<rmm::device_uvector<size_type>>(std::move(unprocessed_matches_per_row)),
          std::move(preprocessed_left));
      }
      return std::make_unique<sort_merge_join_match_context>(
        left, std::move(matches_per_row), std::move(preprocessed_left));
    },
    stream);
}

}  // namespace cudf::detail
