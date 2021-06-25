/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <thrust/optional.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Struct which contains per-column information necessary to
 * traverse a column hierarchy on the gpu.
 *
 * When `row_bit_count` is called, the input column hierarchy is flattened into a
 * vector of column_device_views.  For each one of them, we store a column_info
 * struct.   The `depth` field represents the depth of the column in the original
 * hierarchy.
 *
 * As we traverse the hierarchy for each input row, we maintain a span representing
 * the start and end rows for the current nesting depth.  At depth 0, this span is
 * always just 1 row.  As we cross list boundaries int the hierarchy, this span
 * grows. So for each column we visit we always know how many rows of it are relevant
 * and can compute it's contribution to the overall size.
 *
 *  An example using a list<list<int>> column, computing the size of row 1.
 *
 *  { {{1, 2}, {3, 4}, {5, 6}}, {{7}, {8, 9, 10}, {11, 12, 13, 14}} }
 *
 *  L0 = List<List<int32_t>>:
 *  Length : 2
 *  Offsets : 0, 3, 6
 *     L1 = List<int32_t>:
 *     Length : 6
 *     Offsets : 0, 2, 4, 6, 7, 10, 14
 *        I = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
 *
 *
 *  span0 = [1, 2]                                        row 1 is represented by the span [1, 2]
 *  span1 = [L0.offsets[span0[0]], L0.offsets[span0[1]]]  expand by the offsets of L0
 *  span1 = [3, 6]                                        span applied to children of L0
 *  span2 = [L1.offsets[span1[0]], L1.offsets[span1[1]]]  expand by the offsets of L1
 *  span2 = [6, 14]                                       span applied to children of L1
 *
 *  The total size of our row is computed as:
 *  (span0[1] - span0[0]) * sizeof(int)        the cost of the offsets for L0
 *                 +
 *  (span1[1] - span1[0]) * sizeof(int)        the cost of the offsets for L1
 *                 +
 *  (span2[1] - span2[0]) * sizeof(int)        the cost of the integers in I
 *
 * `depth` represents our depth in the source column hierarchy.
 *
 * "branches" within the spans can occur when we have lists inside of structs.
 * consider a case where we are entering a struct<list, float> with a span of [4, 8].
 * The internal list column will change that span to something else, say [5, 9].
 * But when we finish processing the list column, the final float column wants to
 * go back and use the original span [4, 8].
 *
 * [4, 8]  [5, 9]   [4, 8]
 * struct< list<>   float>
 *
 * To accomplish this we maintain a stack of spans. Pushing the current span
 * whenever we enter a branch, and popping a span whenever we leave a branch.
 *
 * `branch_depth_start` represents the branch depth as we reach a new column.
 * if `branch_depth_start` is < the last branch depth we saw, we are returning
 * from a branch and should pop off the stack.
 *
 * `branch_depth_end` represents the new branch depth caused by this column.
 * if branch_depth_end > branch_depth_start, we are branching and need to
 * push the current span on the stack.
 *
 */
struct column_info {
  size_type depth;
  size_type branch_depth_start;
  size_type branch_depth_end;
};

/**
 * @brief Struct which contains hierarchy information precomputed on the host.
 *
 * If the input data contains only fixed-width types, this preprocess step
 * produces the value `simple_per_row_size` which is a constant for every
 * row in the output.  We can use this value and skip the more complicated
 * processing for lists, structs and strings entirely if `complex_type_count`
 * is 0.
 *
 */
struct hierarchy_info {
  hierarchy_info() : simple_per_row_size(0), complex_type_count(0), max_branch_depth(0) {}

  // These two fields act as an optimization. If we find that the entire table
  // is just fixed-width types, we do not need to do the more expensive kernel call that
  // traverses the individual columns. So if complex_type_count is 0, we can just
  // return a column where every row contains the value simple_per_row_size
  size_type simple_per_row_size;  // in bits
  size_type complex_type_count;

  // max depth of span branches present in the hierarchy.
  size_type max_branch_depth;
};

/**
 * @brief Function which flattens the incoming column hierarchy into a vector
 * of column_views and produces accompanying column_info and hierarchy_info
 * metadata.
 *
 * @param begin: Beginning of a range of column views
 * @param end: End of a range of column views
 * @param out: (output) Flattened vector of output column_views
 * @param info: (output) Additional per-output column_view metadata needed by the gpu
 * @param h_info: (output) Information about the hierarchy
 * @param cur_depth: Current absolute depth in the hierarchy
 * @param cur_branch_depth: Current branch depth
 * @param parent_index: Index into `out` representing our owning parent column
 */
template <typename ColIter>
void flatten_hierarchy(ColIter begin,
                       ColIter end,
                       std::vector<cudf::column_view>& out,
                       std::vector<column_info>& info,
                       hierarchy_info& h_info,
                       rmm::cuda_stream_view stream,
                       size_type cur_depth                = 0,
                       size_type cur_branch_depth         = 0,
                       thrust::optional<int> parent_index = {});

/**
 * @brief Type-dispatched functor called by flatten_hierarchy.
 *
 */
struct flatten_functor {
  rmm::cuda_stream_view stream;

  // fixed width
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(column_view const& col,
                  std::vector<cudf::column_view>& out,
                  std::vector<column_info>& info,
                  hierarchy_info& h_info,
                  rmm::cuda_stream_view,
                  size_type cur_depth,
                  size_type cur_branch_depth,
                  thrust::optional<int>)
  {
    out.push_back(col);
    info.push_back({cur_depth, cur_branch_depth, cur_branch_depth});
    h_info.simple_per_row_size +=
      (sizeof(device_storage_type_t<T>) * CHAR_BIT) + (col.nullable() ? 1 : 0);
  }

  // strings
  template <typename T, std::enable_if_t<std::is_same<T, string_view>::value>* = nullptr>
  void operator()(column_view const& col,
                  std::vector<cudf::column_view>& out,
                  std::vector<column_info>& info,
                  hierarchy_info& h_info,
                  rmm::cuda_stream_view,
                  size_type cur_depth,
                  size_type cur_branch_depth,
                  thrust::optional<int>)
  {
    out.push_back(col);
    info.push_back({cur_depth, cur_branch_depth, cur_branch_depth});
    h_info.complex_type_count++;
  }

  // lists
  template <typename T, std::enable_if_t<std::is_same<T, list_view>::value>* = nullptr>
  void operator()(column_view const& col,
                  std::vector<cudf::column_view>& out,
                  std::vector<column_info>& info,
                  hierarchy_info& h_info,
                  rmm::cuda_stream_view stream,
                  size_type cur_depth,
                  size_type cur_branch_depth,
                  thrust::optional<int> parent_index)
  {
    // track branch depth as we reach this list and after we pass it
    size_type const branch_depth_start = cur_branch_depth;
    auto const is_list_inside_struct =
      parent_index && out[parent_index.value()].type().id() == type_id::STRUCT;
    if (is_list_inside_struct) {
      cur_branch_depth++;
      h_info.max_branch_depth = max(h_info.max_branch_depth, cur_branch_depth);
    }
    size_type const branch_depth_end = cur_branch_depth;

    out.push_back(col);
    info.push_back({cur_depth, branch_depth_start, branch_depth_end});

    lists_column_view lcv(col);
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [col = lcv.get_sliced_child(stream)](auto) { return col; });
    h_info.complex_type_count++;

    flatten_hierarchy(
      iter, iter + 1, out, info, h_info, stream, cur_depth + 1, cur_branch_depth, out.size() - 1);
  }

  // structs
  template <typename T, std::enable_if_t<std::is_same<T, struct_view>::value>* = nullptr>
  void operator()(column_view const& col,
                  std::vector<cudf::column_view>& out,
                  std::vector<column_info>& info,
                  hierarchy_info& h_info,
                  rmm::cuda_stream_view stream,
                  size_type cur_depth,
                  size_type cur_branch_depth,
                  thrust::optional<int>)
  {
    out.push_back(col);
    info.push_back({cur_depth, cur_branch_depth, cur_branch_depth});

    h_info.simple_per_row_size += col.nullable() ? 1 : 0;

    structs_column_view scv(col);
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [&scv](auto i) { return scv.get_sliced_child(i); });
    flatten_hierarchy(iter,
                      iter + scv.num_children(),
                      out,
                      info,
                      h_info,
                      stream,
                      cur_depth + 1,
                      cur_branch_depth,
                      out.size() - 1);
  }

  // everything else
  template <typename T, typename... Args>
  std::enable_if_t<!cudf::is_fixed_width<T>() && !std::is_same<T, string_view>::value &&
                     !std::is_same<T, list_view>::value && !std::is_same<T, struct_view>::value,
                   void>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported column type in row_bit_count");
  }
};

template <typename ColIter>
void flatten_hierarchy(ColIter begin,
                       ColIter end,
                       std::vector<cudf::column_view>& out,
                       std::vector<column_info>& info,
                       hierarchy_info& h_info,
                       rmm::cuda_stream_view stream,
                       size_type cur_depth,
                       size_type cur_branch_depth,
                       thrust::optional<int> parent_index)
{
  std::for_each(begin, end, [&](column_view const& col) {
    cudf::type_dispatcher(col.type(),
                          flatten_functor{stream},
                          col,
                          out,
                          info,
                          h_info,
                          stream,
                          cur_depth,
                          cur_branch_depth,
                          parent_index);
  });
}

/**
 * @brief Struct representing a span of rows.
 *
 */
struct row_span {
  size_type row_start, row_end;
};

/**
 * @brief Functor for computing the size, in bits, of a `row_span` of rows for a given
 * `column_device_view`
 *
 */
struct row_size_functor {
  /**
   * @brief Computes size in bits of a span of rows in a fixed-width column.
   *
   * Computed as :   ((# of rows) * sizeof(data type) * 8)
   *                 +
   *                 1 bit per row for validity if applicable.
   */
  template <typename T>
  __device__ size_type operator()(column_device_view const& col, row_span const& span)
  {
    auto const num_rows{span.row_end - span.row_start};
    auto const element_size  = sizeof(device_storage_type_t<T>) * CHAR_BIT;
    auto const validity_size = col.nullable() ? 1 : 0;
    return (element_size + validity_size) * num_rows;
  }
};

/**
 * @brief Computes size in bits of a span of rows in a strings column.
 *
 * Computed as :   ((# of rows) * sizeof(offset) * 8) + (total # of characters * 8))
 *                 +
 *                 1 bit per row for validity if applicable.
 */
template <>
__device__ size_type row_size_functor::operator()<string_view>(column_device_view const& col,
                                                               row_span const& span)
{
  column_device_view const& offsets = col.child(strings_column_view::offsets_column_index);
  auto const num_rows{span.row_end - span.row_start};
  auto const row_start{span.row_start + col.offset()};
  auto const row_end{span.row_end + col.offset()};

  auto const offsets_size  = sizeof(offset_type) * CHAR_BIT;
  auto const validity_size = col.nullable() ? 1 : 0;
  auto const chars_size =
    (offsets.data<offset_type>()[row_end] - offsets.data<offset_type>()[row_start]) * CHAR_BIT;
  return ((offsets_size + validity_size) * num_rows) + chars_size;
}

/**
 * @brief Computes size in bits of a span of rows in a list column.
 *
 * Computed as :   ((# of rows) * sizeof(offset) * 8)
 *                 +
 *                 1 bit per row for validity if applicable.
 */
template <>
__device__ size_type row_size_functor::operator()<list_view>(column_device_view const& col,
                                                             row_span const& span)
{
  auto const num_rows{span.row_end - span.row_start};

  auto const offsets_size  = sizeof(offset_type) * CHAR_BIT;
  auto const validity_size = col.nullable() ? 1 : 0;
  return (offsets_size + validity_size) * num_rows;
}

/**
 * @brief Computes size in bits of a span of rows in a struct column.
 *
 * Computed as :   1 bit per row for validity if applicable.
 */
template <>
__device__ size_type row_size_functor::operator()<struct_view>(column_device_view const& col,
                                                               row_span const& span)
{
  auto const num_rows{span.row_end - span.row_start};
  return (col.nullable() ? 1 : 0) * num_rows;  // cost of validity
}

/**
 * @brief Kernel for computing per-row sizes in bits.
 *
 * @param cols An span of column_device_views represeting a column hierarcy
 * @param info An span of column_info structs corresponding the elements in `cols`
 * @param output Output span of size (# rows) where per-row bit sizes are stored
 * @param max_branch_depth Maximum depth of the span stack needed per-thread
 */
__global__ void compute_row_sizes(device_span<column_device_view const> cols,
                                  device_span<column_info const> info,
                                  device_span<size_type> output,
                                  size_type max_branch_depth)
{
  extern __shared__ row_span thread_branch_stacks[];
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto const num_rows = output.size();
  if (tid >= num_rows) { return; }

  // branch stack. points to the last list prior to branching.
  row_span* my_branch_stack = thread_branch_stacks + (tid * max_branch_depth);
  size_type branch_depth{0};

  // current row span - always starts at 1 row.
  row_span cur_span{tid, tid + 1};

  // output size
  size_type& size = output[tid];
  size            = 0;

  size_type last_branch_depth{0};
  for (size_type idx = 0; idx < cols.size(); idx++) {
    column_device_view const& col = cols[idx];

    // if we've returned from a branch
    if (info[idx].branch_depth_start < last_branch_depth) {
      cur_span = my_branch_stack[--branch_depth];
    }
    // if we're entering a new branch.
    // NOTE: this case can happen (a pop and a push by the same column)
    // when we have a struct<list, list>
    if (info[idx].branch_depth_end > info[idx].branch_depth_start) {
      my_branch_stack[branch_depth++] = cur_span;
    }

    // if we're back at depth 0, this is a new top-level column, so reset
    // span info
    if (info[idx].depth == 0) {
      branch_depth      = 0;
      last_branch_depth = 0;
      cur_span          = row_span{tid, tid + 1};
    }

    // add the contributing size of this row
    size += cudf::type_dispatcher(col.type(), row_size_functor{}, col, cur_span);

    // if this is a list column, update the working span from our offsets
    if (col.type().id() == type_id::LIST) {
      column_device_view const& offsets = col.child(lists_column_view::offsets_column_index);
      auto const base_offset            = offsets.data<offset_type>()[col.offset()];
      cur_span.row_start =
        offsets.data<offset_type>()[cur_span.row_start + col.offset()] - base_offset;
      cur_span.row_end = offsets.data<offset_type>()[cur_span.row_end + col.offset()] - base_offset;
    }

    last_branch_depth = info[idx].branch_depth_end;
  }
}

}  // anonymous namespace

/**
 * @copydoc cudf::detail::row_bit_count
 *
 */
std::unique_ptr<column> row_bit_count(table_view const& t,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // no rows
  if (t.num_rows() <= 0) { return cudf::make_empty_column(data_type{type_id::INT32}); }

  // flatten the hierarchy and determine some information about it.
  std::vector<cudf::column_view> cols;
  std::vector<column_info> info;
  hierarchy_info h_info;
  flatten_hierarchy(t.begin(), t.end(), cols, info, h_info, stream);
  CUDF_EXPECTS(info.size() == cols.size(), "Size/info mismatch");

  // create output buffer and view
  auto output = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, t.num_rows(), mask_state::UNALLOCATED, stream, mr);
  mutable_column_view mcv = output->mutable_view();

  // simple case.  if we have no complex types (lists, strings, etc), the per-row size is already
  // trivially computed
  if (h_info.complex_type_count <= 0) {
    thrust::fill(rmm::exec_policy(stream),
                 mcv.begin<size_type>(),
                 mcv.end<size_type>(),
                 h_info.simple_per_row_size);
    return output;
  }

  // create a contiguous block of column_device_views
  auto d_cols = contiguous_copy_column_device_views<column_device_view>(cols, stream);

  // move stack info to the gpu
  rmm::device_uvector<column_info> d_info(info.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(d_info.data(),
                           info.data(),
                           sizeof(column_info) * info.size(),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  // each thread needs to maintain a stack of row spans of size max_branch_depth. we will use
  // shared memory to do this rather than allocating a potentially gigantic temporary buffer
  // of memory of size (# input rows * sizeof(row_span) * max_branch_depth).
  auto const shmem_per_thread = sizeof(row_span) * h_info.max_branch_depth;
  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  constexpr int max_block_size = 256;
  auto const block_size =
    shmem_per_thread != 0
      ? std::min(max_block_size, shmem_limit_per_block / static_cast<int>(shmem_per_thread))
      : max_block_size;
  auto const shared_mem_size = shmem_per_thread * block_size;
  // should we be aborting if we reach some extremely small block size, or just if we hit 0?
  CUDF_EXPECTS(block_size > 0, "Encountered a column hierarchy too complex for row_bit_count");

  cudf::detail::grid_1d grid{t.num_rows(), block_size, 1};
  compute_row_sizes<<<grid.num_blocks, block_size, shared_mem_size, stream.value()>>>(
    {std::get<1>(d_cols), cols.size()},
    {d_info.data(), info.size()},
    {mcv.data<size_type>(), static_cast<std::size_t>(t.num_rows())},
    h_info.max_branch_depth);

  return output;
}

}  // namespace detail

/**
 * @copydoc cudf::row_bit_count
 *
 */
std::unique_ptr<column> row_bit_count(table_view const& t, rmm::mr::device_memory_resource* mr)
{
  return detail::row_bit_count(t, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
