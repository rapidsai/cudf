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
    
    // make_validity_iterator returns a boolean iterator that sums to 1
    // so we need to transform it to cast it to an integer type
    auto bitmask_iterator = thrust::make_transform_iterator(
      experimental::detail::make_validity_iterator(*values_view),
      [] __device__ (auto i) -> size_type { return i; });

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
