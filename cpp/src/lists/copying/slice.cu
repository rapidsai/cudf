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
  lists_count = end - start;

  auto execpol = rmm::exec_policy(stream);

  // account for the offset of the view:
  start += lists.offset();
  end += lists.offset();

  // offsets at the beginning and end of the slice:
  cudf::size_type start_offset{0};
  cudf::size_type end_offset{0};
  auto offsets_data = lists.offsets().data<cudf::size_type>();
  CUDA_TRY(cudaMemcpyAsync(
    &start_offset, offsets_data + start, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaMemcpyAsync(
    &end_offset, offsets_data + end, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream));

  // compute the offsets column of the result:
  rmm::device_uvector<cudf::size_type> out_offsets(lists_count + 1, stream);

  thrust::transform(execpol->on(stream),
                    offsets_data + start,
                    offsets_data + end + 1,
                    out_offsets.data(),
                    [start_offset] __device__(cudf::size_type i) { return i - start_offset; });
  auto offsets = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, lists_count + 1, out_offsets.release());

  // compute the child column of the result:
  auto child =
    (lists.child().type() == cudf::data_type{type_id::LIST})
      ? slice(lists_column_view(lists.child()), start_offset, end_offset, stream, mr)
      : std::make_unique<cudf::column>(
          cudf::detail::slice(lists.child(), {start_offset, end_offset}, stream).front());

  // compute the null mask of the result:
  auto null_mask = cudf::copy_bitmask(lists.null_mask(), start_offset, end_offset, stream, mr);

  return make_lists_column(lists_count,
                           std::move(offsets),
                           std::move(child),
                           cudf::UNKNOWN_NULL_COUNT,
                           std::move(null_mask));
}
}  // namespace detail
}  // namespace lists
}  // namespace cudf
