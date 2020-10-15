#include "csv_test_new.cuh"
#include "rmm/thrust_rmm_allocator.h"

template <typename T_,
          typename ScanOp_,
          typename PredicateOp_,
          int BLOCK_DIM_X_,
          int ITEMS_PER_THREAD_>
struct InclusiveScanCopyIfPolicy {
  static constexpr int BLOCK_DIM_X      = BLOCK_DIM_X_;
  static constexpr int ITEMS_PER_THREAD = ITEMS_PER_THREAD_;
  using T                               = T_;
  using ScanOp                          = ScanOp_;
  using PredicateOp                     = PredicateOp_;
};

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_scan_1(  //
  device_span<T> input,
  device_span<T> block_temp_values,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_scan_2(  //
  device_span<T> block_temp_values,
  ScanOp scan_op)
{
}

template <typename Policy,
          typename T           = typename Policy::T,
          typename ScanOp      = typename Policy::ScanOp,
          typename PredicateOp = typename Policy::PredicateOp,
          int BLOCK_DIM_X      = Policy::BLOCK_DIM_X,
          int ITEMS_PER_THREAD = Policy::ITEMS_PER_THREAD>
__global__ void kernel_pass_gather(  //
  device_span<T> input,
  device_span<uint32_t> output,
  device_span<T> block_temp_value,
  device_span<uint32_t> block_temp_count,
  ScanOp scan_op,
  PredicateOp predicate_op)
{
}

/**
 * @brief inclusive_scan + copy_if
 *
 * f(a)    -> b // upgrade input to state
 * f(b, a) -> b // integrate input to state
 * f(b, b) -> b // merge state with state
 * f(b)    -> c // downgrade state to output
 *
 * @tparam T
 * @tparam ScanOp
 * @tparam PredicateOp
 * @param input
 * @param scan_op
 * @param predicate_op
 * @param stream
 * @return rmm::device_vector<uint32_t>
 */
template <typename T,
          typename ScanOp,
          typename PredicateOp>
rmm::device_vector<uint32_t>  //
inclusive_scan_copy_if(device_span<T> input,
                       ScanOp scan_op,
                       PredicateOp predicate_op,
                       cudaStream_t stream = 0)
{
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 8 };  // 1b * 1t * 8i : [pass]
  // enum { BLOCK_DIM_X = 8, ITEMS_PER_THREAD = 1 };  // 1b * 8t * 1i : [pass]
  // enum { BLOCK_DIM_X = 1, ITEMS_PER_THREAD = 4 };  // 2b * 1t * 4i : [fail]
  enum { BLOCK_DIM_X = 2, ITEMS_PER_THREAD = 2 };  // 2b * 2t * 2i : [fail]

  using Policy = InclusiveScanCopyIfPolicy<T, ScanOp, PredicateOp, BLOCK_DIM_X, ITEMS_PER_THREAD>;

  cudf::detail::grid_1d grid(input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);

  auto d_block_temp_values = rmm::device_vector<T>(grid.num_blocks + 1);
  auto d_block_temp_counts = rmm::device_vector<uint32_t>(grid.num_blocks + 1);
  auto kernel_scan_1       = kernel_pass_scan_1<Policy>;
  auto kernel_scan_2       = kernel_pass_scan_2<Policy>;
  auto kernel_gather       = kernel_pass_gather<Policy>;

  // block-wise aggregates
  kernel_scan_1<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    d_block_temp_values,
    scan_op);

  // device-wise aggregates
  kernel_scan_2<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    d_block_temp_values,
    scan_op);

  // device-wise count
  kernel_gather<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    device_span<uint32_t>(nullptr, 0),
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  // device-wise gather
  auto output = rmm::device_vector<uint32_t>(d_block_temp_counts.back());
  kernel_gather<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    input,
    output,
    d_block_temp_values,
    d_block_temp_counts,
    scan_op,
    predicate_op);

  return output;
}
