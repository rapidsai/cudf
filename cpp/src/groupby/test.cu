// test.cu

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {

auto test(column_view const& values,
          size_type num_groups,
          cudf::device_span<cudf::size_type const> group_labels,
          rmm::cuda_stream_view stream,
          rmm::mr::device_memory_resource* mr)
{
  auto result = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, 1, mask_state::UNALLOCATED, stream, mr);

  auto const d_flattened_values_ptr = table_device_view::create(table_view{{values}}, stream);
  auto const count_iter             = thrust::make_counting_iterator<size_type>(0);
  auto const result_begin           = result->mutable_view().template begin<size_type>();

  {
    auto const binop =
      row_lexicographic_comparator<true>(*d_flattened_values_ptr, *d_flattened_values_ptr);
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.data(),
                          group_labels.data() + group_labels.size(),
                          count_iter,
                          thrust::make_discard_iterator(),
                          result_begin,
                          thrust::equal_to<size_type>{},
                          binop);
  }


  {
    auto const binop =
      row_lexicographic_comparator<false>(*d_flattened_values_ptr, *d_flattened_values_ptr);
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.data(),
                          group_labels.data() + group_labels.size(),
                          count_iter,
                          thrust::make_discard_iterator(),
                          result_begin,
                          thrust::equal_to<size_type>{},
                          binop);
  }

  return result;
}

}  // namespace cudf
