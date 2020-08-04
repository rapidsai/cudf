#include <thrust/iterator/counting_iterator.h>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <iostream>

namespace cudf {
namespace lists {
namespace detail {

// new lists column from a subset of a lists_column_view
std::unique_ptr<cudf::column> slice(lists_column_view const& lists,
                                    size_type start,
                                    size_type end,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type lists_count = lists.size();
  if (lists_count == 0) { return cudf::empty_like(lists.parent()); }

  if (end < 0 || end > lists_count) end = lists_count;
  CUDF_EXPECTS(((start >= 0) && (start < end)), "Invalid start parameter value.");

  auto execpol = rmm::exec_policy(stream);

  lists_count = end - start;
  cudf::size_type start_offset{0};
  cudf::size_type end_offset{0};
  auto offsets = lists.offsets().data<cudf::size_type>();

  CUDA_TRY(cudaMemcpyAsync(
    &start_offset, offsets + start, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaMemcpyAsync(
    &end_offset, offsets + end, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream));

  rmm::device_uvector<cudf::size_type> out_offsets(lists_count + 1, stream);

  thrust::transform(execpol->on(stream),
                    offsets + start,
                    offsets + end + 1,
                    out_offsets.data(),
                    [start_offset] __device__(cudf::size_type i) { return i - start_offset; });

  auto offsets_column = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, lists_count + 1, out_offsets.release());

  std::unique_ptr<cudf::column> child_column =
    (lists.child().type() == cudf::data_type{type_id::LIST})
      ? slice(lists_column_view(lists.child()), start_offset, end_offset, stream, mr)
      : std::make_unique<cudf::column>(lists.child(), stream, mr);

  auto null_mask = cudf::copy_bitmask(lists.null_mask(), start_offset, end_offset, stream, mr);

  return make_lists_column(lists_count,
                           std::move(offsets_column),
                           std::move(child_column),
                           cudf::UNKNOWN_NULL_COUNT,
                           std::move(null_mask));
}
}  // namespace detail
}  // namespace lists
}  // namespace cudf
