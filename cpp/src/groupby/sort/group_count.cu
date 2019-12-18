#include "group_reductions.hpp"

#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {


std::unique_ptr<column> group_count(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  auto result = make_numeric_column(data_type(type_to_id<size_type>()),
                  num_groups, mask_state::UNALLOCATED, stream, mr);

  if (values.nullable()) {
    auto values_view = column_device_view::create(values);
    
    auto bitmask_iterator = experimental::detail::make_validity_iterator(
                                                    *values_view);

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          bitmask_iterator,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  } else {
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_labels.begin(),
                          group_labels.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<size_type>());
  }

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
