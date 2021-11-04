// test.cu

#if 0
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

#else

#include <cudf/table/row_operators.cuh>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace cudf {
namespace groupby {
namespace detail {
template <bool has_nulls>
struct row_arg_minmax_fn {
  size_type const num_rows;
  row_lexicographic_comparator<has_nulls> const comp;
  bool const arg_min;

  row_arg_minmax_fn(size_type const num_rows_,
                    table_device_view const& table_,
                    null_order const* null_precedence_,
                    bool const arg_min_)
    : num_rows(num_rows_), comp(table_, table_, nullptr, null_precedence_), arg_min(arg_min_)
  {
  }

  CUDA_DEVICE_CALLABLE auto operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    if (lhs_idx < 0 || lhs_idx >= num_rows) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= num_rows) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    return comp(lhs_idx, rhs_idx) == arg_min ? lhs_idx : rhs_idx;
  }
};

std::unique_ptr<column> group_argminmax_struct(aggregation::Kind K,
                                               column_view const& values,
                                               size_type num_groups,
                                               cudf::device_span<size_type const> group_labels,
                                               column_view const& key_sort_order,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(K == aggregation::ARGMIN || K == aggregation::ARGMAX,
               "Only groupby ARGMIN/ARGMAX are supported for STRUCT type.");

  auto result = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_groups, mask_state::UNALLOCATED, stream, mr);

  if (values.is_empty()) { return result; }

  // When finding ARGMIN, we need to consider nulls as larger than non-null elements.
  // Thing is opposite for ARGMAX.
  auto const null_precedence  = (K == aggregation::ARGMIN) ? null_order::AFTER : null_order::BEFORE;
  auto const flattened_values = structs::detail::flatten_nested_columns(
    table_view{{values}}, {}, std::vector<null_order>{null_precedence});
  auto const d_flattened_values_ptr = table_device_view::create(flattened_values, stream);
  auto const flattened_null_precedences =
    (K == aggregation::ARGMIN)
      ? cudf::detail::make_device_uvector_async(flattened_values.null_orders(), stream)
      : rmm::device_uvector<null_order>(0, stream);

  // Perform segmented reduction to find ARGMIN/ARGMAX.
  auto const do_reduction = [&](auto const& inp_iter, auto const& out_iter, auto const& binop) {
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          group_labels.data(),
                          group_labels.data() + group_labels.size(),
                          inp_iter,
                          thrust::make_discard_iterator(),
                          out_iter,
                          thrust::equal_to<size_type>{},
                          binop);
  };

  auto const count_iter   = thrust::make_counting_iterator<size_type>(0);
  auto const result_begin = result->mutable_view().template begin<size_type>();
  if (values.has_nulls()) {
    auto const binop = row_arg_minmax_fn<true>(values.size(),
                                               *d_flattened_values_ptr,
                                               flattened_null_precedences.data(),
                                               K == aggregation::ARGMIN);
    do_reduction(count_iter, result_begin, binop);

    // Generate bitmask for the output by segmented reduction of the input bitmask.
    auto const d_values_ptr = column_device_view::create(values, stream);
    auto validity           = rmm::device_uvector<bool>(num_groups, stream);
    do_reduction(cudf::detail::make_validity_iterator(*d_values_ptr),
                 validity.begin(),
                 thrust::logical_or<bool>{});

    auto [null_mask, null_count] = cudf::detail::valid_if(
      validity.begin(), validity.end(), thrust::identity<bool>{}, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
  } else {
    auto const binop = row_arg_minmax_fn<false>(values.size(),
                                                *d_flattened_values_ptr,
                                                flattened_null_precedences.data(),
                                                K == aggregation::ARGMIN);
    do_reduction(count_iter, result_begin, binop);
  }

  // result now stores the indices of minimum elements in the sorted values.
  // We need the indices of minimum elements in the original unsorted values.
  thrust::gather(rmm::exec_policy(stream),
                 result_begin,
                 result_begin + num_groups,
                 key_sort_order.template begin<size_type>(),
                 result_begin);

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

#endif
