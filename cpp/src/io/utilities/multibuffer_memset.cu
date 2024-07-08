#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>


#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>

struct memset_task {
  uint64_t size;
  uint64_t * data;
} typedef memset_task;

// 1 task == 1 block
__global__ void memset_kernel(memset_task * tasks, int8_t const value)
{
  auto task = tasks[blockIdx.x];

  // block stride over task.begin, task.end
  auto buf = task.data;
  uint64_t const end = task.size;
  uint64_t memsets_left = (task.size + blockDim.x - 1) / blockDim.x;
  uint64_t t = threadIdx.x;
  while(memsets_left > 0){
    if (t < end) {
      buf[t] = value;
    }
    t += blockDim.x;
    memsets_left -= 1;
  }
}

void multibuffer_memset(std::vector<cudf::device_span<uint8_t>> & bufs, 
                        int8_t const value,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref temp_mr
                        )
{ 
  // define task and bytes paramters
  constexpr uint64_t bytes_per_task = 128 * 1024;
  constexpr uint64_t threads_per_block = 256;
  auto const num_bufs = bufs.size();

  // declare offsets vector which stores for each buffer at which task in the task lists it starts at 
  rmm::device_uvector<std::size_t> offsets(bufs.size() + 1, stream, temp_mr); 

  // copy bufs into gpu and then get sizes from there (cudf detail function make device vector async)
  auto gpu_bufs = cudf::detail::make_device_uvector_async(bufs, stream, temp_mr);

  // get a vector with the sizes of all buffers
  auto buf_count_iter = cudf::detail::make_counting_transform_iterator(
    0, 
    cuda::proclaim_return_type<std::size_t>(
      [gpu_bufs = gpu_bufs.data(), bytes_per_task = bytes_per_task, num_bufs] __device__(cudf::size_type i) {
        size_t temp = cudf::util::round_up_safe(gpu_bufs[i].size(), bytes_per_task);
        return i >= num_bufs ? 0 : temp / bytes_per_task;
      }
    )
  );

  // fill up offsets buffer using exclusive scan
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr), buf_count_iter, buf_count_iter + bufs.size() + 1, offsets.begin(), 0);

  // the total number of tasks is the last number in the offsets vector
  size_t const total_tasks = offsets.back_element(stream);

  // declaring list of tasks to pass onto cuda kernel
  rmm::device_uvector<memset_task> tasks(total_tasks, stream, temp_mr); 

  // fill up task lists based on buffer values
  thrust::transform(
    rmm::exec_policy(stream, temp_mr),
    thrust::make_counting_iterator<std::size_t>(0), 
    thrust::make_counting_iterator<std::size_t>(total_tasks), 
    tasks.begin(), 
    cuda::proclaim_return_type<memset_task>(
      [offsets = offsets.data(), offset_size = num_bufs + 1, gpu_bufs = gpu_bufs.data(), bytes_per_task] __device__(cudf::size_type task) {
        auto buf_idx = thrust::upper_bound(thrust::seq, offsets, offsets + offset_size, task);
        if (*buf_idx > task) {buf_idx -= 1;}
        size_t start = (task - *buf_idx) * bytes_per_task;
        size_t end = (start + bytes_per_task) <= gpu_bufs[buf_idx - offsets].size() ? start + bytes_per_task : gpu_bufs[buf_idx - offsets].size();
        memset_task ret;
        ret.size = (end - start) / 8;
        ret.data = (uint64_t *)(gpu_bufs[buf_idx - offsets].data() + start);
        return ret;
      }
    )
  );

  // launch cuda kernel
  if (total_tasks != 0) {
    memset_kernel<<<total_tasks, threads_per_block>>>(tasks.data(), value);
  }

}
