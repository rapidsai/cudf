#include "group_reductions.hpp"


namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_sum(
    column_view const& values,
    rmm::device_vector<size_type> const& group_labels,
    size_type num_groups,
    cudaStream_t stream)
{
  return type_dispatcher(values.type(), reduce_functor<aggregation::SUM>{},
                         values, group_labels, num_groups, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
