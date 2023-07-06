/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <nvtext/detail/generate_ngrams.hpp>
#include <nvtext/jaccard.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/reduction/detail/segmented_reduction.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <lists/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Compute the jaccard distance for each row
 *
 * Formula is J = |A ∩ B| / |A ∪ B|
 *              = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 * where |A ∩ B| is number of common values between A and B
 * and |x| is the number of unique values in x.
 *
 * https://en.wikipedia.org/wiki/Jaccard_index
 */
struct jaccard_fn {
  cudf::size_type const* d_offsets1;
  cudf::size_type const* d_offsets2;
  cudf::size_type const* d_inters;

  __device__ float operator()(cudf::size_type idx)
  {
    auto const size1  = d_offsets1[idx + 1] - d_offsets1[idx];
    auto const size2  = d_offsets2[idx + 1] - d_offsets2[idx];
    auto const inters = d_inters[idx];
    // the intersect values are in both sets so a union count
    // would need to subtract the intersect count from one set
    auto const unions = size1 + size2 - inters;
    return unions ? ((float)inters / (float)unions) : 0.f;
  }
};

/**
 * @brief Compute the number of common values within each row
 */
rmm::device_uvector<cudf::size_type> intersect_counts(cudf::lists_column_view const& lhs,
                                                      cudf::lists_column_view const& rhs,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto const lhs_child  = lhs.child();
  auto const rhs_child  = rhs.child();
  auto const lhs_labels = cudf::lists::detail::generate_labels(lhs, lhs_child.size(), stream, mr);
  auto const rhs_labels = cudf::lists::detail::generate_labels(rhs, rhs_child.size(), stream, mr);
  auto const lhs_table  = cudf::table_view{{lhs_labels->view(), lhs_child}};
  auto const rhs_table  = cudf::table_view{{rhs_labels->view(), rhs_child}};

  auto const nulls_eq = cudf::null_equality::EQUAL;
  auto const nans_eq  = cudf::nan_equality::ALL_EQUAL;
  auto const contained =
    cudf::detail::contains(lhs_table, rhs_table, nulls_eq, nans_eq, stream, mr);

  rmm::device_uvector<cudf::size_type> result(rhs.size(), stream);
  auto sum  = thrust::plus<cudf::size_type>{};
  auto init = cudf::size_type{0};
  cudf::reduction::detail::segmented_reduce(
    contained.begin(), rhs.offsets_begin(), rhs.offsets_end(), result.begin(), sum, init, stream);
  return result;
}
}  // namespace

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    input1.size() == input2.size(), "input columns must be the same size", std::invalid_argument);
  CUDF_EXPECTS(width >= 5,
               "Parameter width should be an integer value of 5 or greater",
               std::invalid_argument);

  auto output_type = cudf::data_type{cudf::type_id::FLOAT32};
  if (input1.is_empty()) { return cudf::make_empty_column(output_type); }

  // build the unique values for each input and then intersect them
  auto const [offsets1, offsets2, inters] = [&] {
    auto const def_mr = rmm::mr::get_current_device_resource();
    // using hash values to reduce memory and speed up distinct and intersect
    // collisions should be minimal for smallish width values
    auto hash1 = hash_character_ngrams(input1, width, stream, def_mr);
    auto hash2 = hash_character_ngrams(input2, width, stream, def_mr);
    auto view1 = cudf::lists_column_view(hash1->view());
    auto view2 = cudf::lists_column_view(hash2->view());

    auto nulls_eq = cudf::null_equality::EQUAL;
    auto nans_eq  = cudf::nan_equality::ALL_EQUAL;
    // remove any duplicates within each row for each input
    hash1 = cudf::lists::detail::distinct(view1, nulls_eq, nans_eq, stream, def_mr);
    hash2 = cudf::lists::detail::distinct(view2, nulls_eq, nans_eq, stream, def_mr);
    view1 = cudf::lists_column_view(hash1->view());
    view2 = cudf::lists_column_view(hash2->view());
    // compute the intersection counts for each row
    auto inters = intersect_counts(view1, view2, stream, def_mr);

    // only the offsets are needed for calculating the unique sizes
    return std::tuple{std::move(hash1->release().children.front()),
                      std::move(hash2->release().children.front()),
                      std::move(inters)};
  }();

  auto const d_offsets1 = offsets1->view().data<cudf::size_type>();
  auto const d_offsets2 = offsets2->view().data<cudf::size_type>();

  auto results = cudf::make_numeric_column(
    output_type, input1.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_results = results->mutable_view().data<float>();

  // compute the jaccard using the unique sizes and the intersect counts
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(results->size()),
                    d_results,
                    jaccard_fn{d_offsets1, d_offsets2, inters.data()});

  if (input1.null_count() || input2.null_count()) {
    auto [null_mask, null_count] =
      cudf::detail::bitmask_and(cudf::table_view({input1.parent(), input2.parent()}), stream, mr);
    results->set_null_mask(null_mask, null_count);
  }

  return results;
}

}  // namespace detail

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::jaccard_index(input1, input2, width, cudf::get_default_stream(), mr);
}

}  // namespace nvtext
