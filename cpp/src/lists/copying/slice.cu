#include <thrust/iterator/counting_iterator.h>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<cudf::column> slice(lists_column_view const& list,
                                    size_type start,
                                    size_type end,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto sliced_table = cudf::detail::gather(table_view{{list.parent()}},
                                           thrust::make_counting_iterator<cudf::size_type>(start),
                                           thrust::make_counting_iterator<cudf::size_type>(end),
                                           false,
                                           mr,
                                           stream)
                        ->release();
  return std::move(sliced_table.front());
}
}  // namespace detail
}  // namespace lists
}  // namespace cudf
