#include "group_reductions.hpp"


namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_sum(
    column_view const& values,
    column_view const& group_sizes,
    rmm::device_vector<size_type> const& group_labels,
    cudaStream_t stream)
{
  return type_dispatcher(values.type(), reduce_functor<aggregation::SUM>{},
                         values, group_sizes, group_labels, stream);
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
