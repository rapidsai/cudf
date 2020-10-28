/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <thrust/iterator/discard_iterator.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/device_uvector.hpp>

#include <numeric>

namespace cudf {
namespace {

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;
inline __device__ size_t _round_up_safe(size_t number_to_round, size_t modulus)
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  return rounded_up;
}

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed with column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a seperate kernel
 * block.
 */
struct src_buf_info {
  cudf::type_id type;
  const int* offsets;        // a pointer to device memory offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  size_t buf_size;   // total size of buffer, including padding
  int num_elements;  // # of elements to be copied
  int element_size;  // size of each element in bytes
  int num_rows;  // # of rows (which may be different from num_elements in the case of validity or
                 // offset buffers)
  int src_row_index;  // row index to start reading from from my associated source buffer
  int dst_offset;     // my offset into the per-partition allocation
  int value_shift;    // amount to shift values down by (for offset buffers)
  int bit_shift;      // # of bits to shift right by (for validity buffers)
};

/**
 * @brief Copy a single buffer of column data, shifting values (for offset columns),
 * and validity (for validity buffers) as necessary.
 *
 * Copies a single partition of a source column buffer to a destination buffer. Shifts
 * element values by value_shift in the case of a buffer of offsets (value_shift will
 * only ever be > 0 in that case).  Shifts elements bitwise by bit_shift in the case of
 * a validity buffer (bif_shift will only ever be > 0 in that case).  This function assumes
 * value_shift and bit_shift will never be > 0 at the same time.
 *
 * This function expects:
 * - src may be a misaligned address
 * - dst must be an aligned address
 *
 * This function always does the ALU work related to value_shift and bit_shift because it is
 * entirely memory-bandwidth bound.
 *
 * @param dst Desination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param num_elements Number of elements to copy
 * @param element_size Size of each element in bytes
 * @param src_row_index Row index to start copying at
 * @param stride Size of the kernel block
 * @param value_shift Shift incoming 4-byte offset values down by this amount
 * @param bit_shift Shift incoming data right by this many bits
 *
 */
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t* __restrict__ _src,
                            int t,
                            int num_elements,
                            int element_size,
                            int src_row_index,
                            uint32_t stride,
                            int value_shift,
                            int bit_shift)
{
  uint8_t* src = _src + (src_row_index * element_size);

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  const size_t num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  const uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
  size_t pos         = t * 16;
  stride *= 16;
  while (pos + 20 <= num_bytes) {
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t*>((src + pos) - ofs);
    uint4 v              = uint4{in32[0], in32[1], in32[2], in32[3]};
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs * 8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs * 8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs * 8 + bit_shift);
      v.w = __funnelshift_r(v.w, in32[4], ofs * 8 + bit_shift);
    }
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos / 16] = v;
    pos += stride;
  }

  // copy trailing bytes
  if (t == 0) {
    size_t remainder = num_bytes < 16 ? num_bytes : 16 + (num_bytes % 16);

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and
    // alignment must be a multiple of 4. value shifting and bit shifting are mututally exclusive
    // and will never both be true at the same time.
    if (value_shift || bit_shift) {
      int idx    = (num_bytes - remainder) / 4;
      uint32_t v = remainder > 0 ? (reinterpret_cast<uint32_t*>(src)[idx] - value_shift) : 0;
      while (remainder) {
        uint32_t next =
          remainder > 0 ? (reinterpret_cast<uint32_t*>(src)[idx + 1] - value_shift) : 0;
        reinterpret_cast<uint32_t*>(dst)[idx] = (v >> bit_shift) | (next << (32 - bit_shift));
        v                                     = next;
        idx++;
        remainder -= 4;
      }
    } else {
      while (remainder) {
        int idx                              = num_bytes - remainder--;
        reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
      }
    }
  }
}

/**
 * @brief Kernel which copies a single buffer from a set of partitioned
 * column buffers.
 *
 * When doing a contiguous_split on X columns comprised of N total internal buffers
 * with M splits, we end up having to copy N*M source/destination buffer pairs.
 * This kernel is arrangedsuch that each block copies 1 source/destination pair.
 * This function retrieves the relevant function and then calls copy_buffer to perform
 * the actual copy.
 *
 * @param num_src_bufs Total number of source buffers (N)
 * @param num_partitions Number of partitions the each source buffer is split into (M)
 * @param src_bufs Input source buffers (N)
 * @param dst_bufs Desination buffers (N*M)
 * @param buf_info Information on the range of values to be copied for each destination buffer.
 *
 * @returns Total offset stack size needed for this range of columns.
 */
__global__ void copy_partition(int num_src_bufs,
                               int num_partitions,
                               uint8_t** src_bufs,
                               uint8_t** dst_bufs,
                               dst_buf_info* buf_info)
{
  int partition_index = blockIdx.x / num_src_bufs;
  int src_buf_index   = blockIdx.x % num_src_bufs;
  int t               = threadIdx.x;
  size_t buf_index    = (partition_index * num_src_bufs) + src_buf_index;
  int num_elements    = buf_info[buf_index].num_elements;
  int element_size    = buf_info[buf_index].element_size;
  int stride          = blockDim.x;
  int src_row_index   = buf_info[buf_index].src_row_index;
  uint8_t* src        = src_bufs[src_buf_index];
  uint8_t* dst        = dst_bufs[partition_index] + buf_info[buf_index].dst_offset;

  // copy, shifting offsets and validity bits as needed
  copy_buffer(dst,
              src,
              t,
              num_elements,
              element_size,
              src_row_index,
              stride,
              buf_info[buf_index].value_shift,
              buf_info[buf_index].bit_shift);
}

// The block of function below are all related:
//
// compute_offset_stack_size()
// setup_src_buf_data()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.  For example, setup_src_buf_info() produces a series of information
// structs that must appear in the same order that setup_src_buf_data() produces
// buffers.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param buf_index Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns.
 */
template <typename InputIter>
size_type compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0);

/**
 * @brief Functor that retrieves the offset stack size needed for a single column.
 *
 * Called by compute_offset_stack_size.  This function will recursively call
 * compute_offset_stack_size in the case of nested types.
 */
struct compute_offset_stack_size_functor {
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  size_type operator()(column_view const& col, int offset_depth)
  {
    //     # of list columns above us    # offsets + validity
    return ((offset_depth) * (1 + (col.nullable() ? 1 : 0))) +
           //     # + local offsets             # chars column
           ((offset_depth + 1) * (1));
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value>* = nullptr>
  size_type operator()(column_view const& col, int offset_depth)
  {
    //     # of list columns above us    # offsets + validity
    //       and ourself
    return ((offset_depth) * (1 + (col.nullable() ? 1 : 0))) +
           //     add children
           compute_offset_stack_size(col.child_begin() + 1, col.child_end(), offset_depth + 1);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::struct_view>::value>* = nullptr>
  size_type operator()(column_view const& col, int offset_depth)
  {
    //     # of list columns above us    # validity
    return ((offset_depth) * (col.nullable() ? 1 : 0)) +
           //     add children
           compute_offset_stack_size(col.child_begin(), col.child_end(), offset_depth);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  size_type operator()(column_view const& col, int offset_depth)
  {
    CUDF_FAIL("Unsupported type");
  }

  template <typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  size_type operator()(column_view const& col, int offset_depth)
  {
    //     # of list columns above us   # of buffers coming from this column
    return offset_depth * (1 + (col.nullable() ? 1 : 0));
  }
};

template <typename InputIter>
size_type compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth)
{
  size_type ret = 0;
  std::for_each(begin, end, [&ret, offset_depth](column_view const& col) {
    ret +=
      cudf::type_dispatcher(col.type(), compute_offset_stack_size_functor{}, col, offset_depth);
  });
  return ret;
}

/**
 * @brief Retrieve all buffers for a range of source columns.
 *
 * Retrieve the individual buffers that make up a range of input columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param buf_index Current buffer index (into out_buf)
 * @param out_buf Array of buffers to be output
 *
 * @returns New working buf_index
 */
template <typename InputIter>
size_type setup_src_buf_data(InputIter begin,
                             InputIter end,
                             size_type buf_index,
                             uint8_t** out_buf);

/**
 * @brief Functor that retrieves the individual buffers comprising a single column.
 *
 * Called by setup_src_buf_data.  This function will recursively call setup_src_buf_data in the case
 * of nested types.
 */
struct setup_src_buf_data_functor {
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  size_type operator()(column_view const& col, size_type buf_index, uint8_t** out_buf)
  {
    strings_column_view scv(col);
    if (col.nullable()) {
      out_buf[buf_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
    out_buf[buf_index++] =
      reinterpret_cast<uint8_t*>(const_cast<size_type*>(scv.offsets().begin<size_type>()));
    out_buf[buf_index++] =
      reinterpret_cast<uint8_t*>(const_cast<int8_t*>(scv.chars().begin<int8_t>()));
    return buf_index;
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value>* = nullptr>
  size_type operator()(column_view const& col, size_type buf_index, uint8_t** out_buf)
  {
    lists_column_view lcv(col);
    if (col.nullable()) {
      out_buf[buf_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
    out_buf[buf_index++] =
      reinterpret_cast<uint8_t*>(const_cast<size_type*>(lcv.offsets().begin<size_type>()));
    return setup_src_buf_data(col.child_begin() + 1, col.child_end(), buf_index, out_buf);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::struct_view>::value>* = nullptr>
  size_type operator()(column_view const& col, size_type buf_index, uint8_t** out_buf)
  {
    if (col.nullable()) {
      out_buf[buf_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
    return setup_src_buf_data(col.child_begin(), col.child_end(), buf_index, out_buf);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  size_type operator()(column_view const& col, size_type buf_index, uint8_t** out_buf)
  {
    CUDF_FAIL("Unsupported type");
  }

  template <typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  size_type operator()(column_view const& col, size_type buf_index, uint8_t** out_buf)
  {
    if (col.nullable()) {
      out_buf[buf_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
    out_buf[buf_index++] = reinterpret_cast<uint8_t*>(const_cast<T*>(col.begin<T>()));
    return buf_index;
  }
};

template <typename InputIter>
size_type setup_src_buf_data(InputIter begin, InputIter end, size_type buf_index, uint8_t** out_buf)
{
  std::for_each(begin, end, [&buf_index, &out_buf](column_view const& col) {
    buf_index =
      cudf::type_dispatcher(col.type(), setup_src_buf_data_functor{}, col, buf_index, out_buf);
  });
  return buf_index;
}

/**
 * @brief Count the total number of source buffers we will be copying
 * from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed with column with validity would be 2 buffers (data, validity).
 * A string column with validity would be 3 buffers (chars, offsets, validity).
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffers for this range of columns
 */
template <typename InputIter>
size_type count_src_bufs(InputIter begin, InputIter end);

/**
 * @brief Functor that computes the number of buffers for a single column.
 *
 * Called by count_src_bufs to compute buffer count for single source column.  This function will
 * recursively call count_src_bufs in the case of nested types.
 */
struct buf_count_functor {
  // strings column : offsets, chars, potentially validity
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return 2 + (col.nullable() ? 1 : 0);
  }

  // nested types : # of children + potentially validity
  template <typename T,
            std::enable_if_t<std::is_same<T, cudf::list_view>::value or
                             std::is_same<T, cudf::struct_view>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return count_src_bufs(col.child_begin(), col.child_end()) + (col.nullable() ? 1 : 0);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  size_type operator()(column_view const& col)
  {
    CUDF_FAIL("Unsupported type");
  }

  // plain types : 1 + potentially validity
  template <typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  size_type operator()(column_view const& col)
  {
    return 1 + (col.nullable() ? 1 : 0);
  }
};

template <typename InputIter>
size_type count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    return cudf::type_dispatcher(col.type(), buf_count_functor{}, col);
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), 0);
}

typedef std::pair<src_buf_info*, size_type> src_buf_output;

/**
 * @brief Computes source buffer information for the copy kernel.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param output Pair containing current source buffer info pointer and an integer
 * representing our current offset nesting depth (how many list levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns new src_buf_output after processing this range of input columns
 */
template <typename InputIter>
src_buf_output setup_source_buf_info(InputIter begin,
                                     InputIter end,
                                     src_buf_info* head,
                                     src_buf_output output,
                                     int parent_offset_index = -1,
                                     int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build info for a single source column.  This function will
 * recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  src_buf_output operator()(column_view const& col,
                            src_buf_info* head,
                            src_buf_output output,
                            int parent_offset_index,
                            int offset_depth)
  {
    strings_column_view scv(col);

    output = maybe_add_null_buffer(col, output, parent_offset_index, offset_depth);

    auto offset_col = output.first;

    output.first->type             = type_id::INT32;  // offsets
    output.first->offsets          = scv.offsets().begin<int>();
    output.first->offset_stack_pos = output.second;
    output.second += offset_depth;
    output.first->parent_offsets_index = parent_offset_index;
    output.first->is_validity          = false;
    output.first++;

    // local offsets apply to the chars
    offset_depth++;
    parent_offset_index = offset_col - head;

    output.first->type             = type_id::INT8;  // chars
    output.first->offsets          = nullptr;
    output.first->offset_stack_pos = output.second;
    output.second += offset_depth;
    output.first->parent_offsets_index = parent_offset_index;
    output.first->is_validity          = false;
    output.first++;

    return output;
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::list_view>::value>* = nullptr>
  src_buf_output operator()(column_view const& col,
                            src_buf_info* head,
                            src_buf_output output,
                            int parent_offset_index,
                            int offset_depth)
  {
    lists_column_view lcv(col);

    output = maybe_add_null_buffer(col, output, parent_offset_index, offset_depth);

    auto offset_col = output.first;

    output.first->type             = type_id::INT32;  // offsets
    output.first->offsets          = lcv.offsets().begin<int>();
    output.first->offset_stack_pos = output.second;
    output.second += offset_depth;
    output.first->parent_offsets_index = parent_offset_index;
    output.first->is_validity          = false;
    output.first++;

    // local offsets apply to the remaining children
    offset_depth++;
    parent_offset_index = offset_col - head;
    return setup_source_buf_info(
      col.child_begin() + 1, col.child_end(), head, output, parent_offset_index, offset_depth);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::struct_view>::value>* = nullptr>
  src_buf_output operator()(column_view const& col,
                            src_buf_info* head,
                            src_buf_output output,
                            int parent_offset_index,
                            int offset_depth)
  {
    output = maybe_add_null_buffer(col, output, parent_offset_index, offset_depth);
    return setup_source_buf_info(
      col.child_begin(), col.child_end(), head, output, parent_offset_index, offset_depth);
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  src_buf_output operator()(column_view const& col,
                            src_buf_info* head,
                            src_buf_output output,
                            int parent_offset_index,
                            int offset_depth)
  {
    CUDF_FAIL("Unsupported type");
  }

  template <typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  src_buf_output operator()(column_view const& col,
                            src_buf_info* head,
                            src_buf_output output,
                            int parent_offset_index,
                            int offset_depth)
  {
    output = maybe_add_null_buffer(col, output, parent_offset_index, offset_depth);

    output.first->type             = col.type().id();
    output.first->offsets          = nullptr;
    output.first->offset_stack_pos = output.second;
    output.second += offset_depth;
    output.first->parent_offsets_index = parent_offset_index;
    output.first->is_validity          = false;
    output.first++;

    return output;
  }

 private:
  src_buf_output maybe_add_null_buffer(column_view const& col,
                                       src_buf_output output,
                                       int parent_offset_index,
                                       int offset_depth)
  {
    if (!col.nullable()) { return output; }
    output.first->type             = type_id::INT32;
    output.first->offsets          = nullptr;
    output.first->offset_stack_pos = output.second;
    output.second += offset_depth;
    output.first->parent_offsets_index = parent_offset_index;
    output.first->is_validity          = true;
    output.first++;

    return output;
  }
};

template <typename InputIter>
src_buf_output setup_source_buf_info(InputIter begin,
                                     InputIter end,
                                     src_buf_info* head,
                                     src_buf_output output,
                                     int parent_offset_index,
                                     int offset_depth)
{
  std::for_each(
    begin, end, [head, &output, parent_offset_index, offset_depth](column_view const& col) {
      output = cudf::type_dispatcher(
        col.type(), buf_info_functor{}, col, head, output, parent_offset_index, offset_depth);
    });
  return output;
}

/**
 * @brief Given a set of input columns and computed split buffers, produce
 * output columns.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param dst Output vector of column views to produce
 * @param partition_index Partition index (which section of the split we are working with)
 * @param buf_index Absolute split output buffer index
 * @param h_dst_bufs Array of destination buffers (indexed by partition_index)
 * @param h_dst_buf_info Array of offsets into destination buffers (indexed by buf_index)
 *
 * @returns new working buffer index after processing this range of input columns
 */
template <typename InputIter>
size_type build_output_columns(InputIter begin,
                               InputIter end,
                               std::vector<column_view>& dst,
                               int partition_index,
                               int buf_index,
                               uint8_t** h_dst_bufs,
                               dst_buf_info* h_dst_buf_info);

/**
 * @brief Functor that builds an output column based on split partition buffers.
 *
 * Called by build_output_columns to build a single output column.  This function will
 * recursively call build_output_columns in the case of nested types.
 */
struct build_column_functor {
  template <typename T, std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  size_type operator()(column_view const& src,
                       std::vector<column_view>& dst,
                       int partition_index,
                       int buf_index,
                       uint8_t** h_dst_bufs,
                       dst_buf_info* h_dst_buf_info)
  {
    bool nullable       = src.nullable();
    int child_buf_index = nullable ? buf_index + 1 : buf_index;

    dst.push_back(cudf::column_view{
      data_type{type_id::STRING},
      h_dst_buf_info[child_buf_index].num_rows,
      nullptr,
      nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[partition_index] +
                                                 h_dst_buf_info[buf_index].dst_offset)
               : nullptr,
      nullable ? UNKNOWN_NULL_COUNT : 0,
      0,
      {cudf::column_view{data_type{type_id::INT32},
                         h_dst_buf_info[child_buf_index].num_elements,
                         reinterpret_cast<void*>(h_dst_bufs[partition_index] +
                                                 h_dst_buf_info[child_buf_index].dst_offset)},

       cudf::column_view{
         data_type{type_id::INT8},
         h_dst_buf_info[child_buf_index + 1].num_elements,
         reinterpret_cast<void*>(h_dst_bufs[partition_index] +
                                 h_dst_buf_info[child_buf_index + 1].dst_offset)}}});
    return buf_index + (nullable ? 3 : 2);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, cudf::struct_view>::value ||
                             std::is_same<T, cudf::list_view>::value>* = nullptr>
  size_type operator()(column_view const& src,
                       std::vector<column_view>& dst,
                       int partition_index,
                       int buf_index,
                       uint8_t** h_dst_bufs,
                       dst_buf_info* h_dst_buf_info)
  {
    bool nullable  = src.nullable();
    int root_index = buf_index;
    if (nullable) { buf_index++; }

    // build children
    std::vector<column_view> children;
    buf_index = build_output_columns(src.child_begin(),
                                     src.child_end(),
                                     children,
                                     partition_index,
                                     buf_index,
                                     h_dst_bufs,
                                     h_dst_buf_info);

    // build me
    dst.push_back(cudf::column_view{
      src.type(),
      h_dst_buf_info[root_index].num_rows,
      nullptr,
      nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[partition_index] +
                                                 h_dst_buf_info[root_index].dst_offset)
               : nullptr,
      nullable ? UNKNOWN_NULL_COUNT : 0,
      0,
      children});
    return buf_index;
  }

  template <typename T, std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  size_type operator()(column_view const& src,
                       std::vector<column_view>& dst,
                       int partition_index,
                       int buf_index,
                       uint8_t** h_dst_bufs,
                       dst_buf_info* h_dst_buf_info)
  {
    CUDF_FAIL("Unsupported type");
  }

  template <typename T, std::enable_if_t<!cudf::is_compound<T>()>* = nullptr>
  size_type operator()(column_view const& src,
                       std::vector<column_view>& dst,
                       int partition_index,
                       int buf_index,
                       uint8_t** h_dst_bufs,
                       dst_buf_info* h_dst_buf_info)
  {
    bool nullable  = src.nullable();
    int root_index = nullable ? buf_index + 1 : buf_index;
    dst.push_back(cudf::column_view{
      src.type(),
      h_dst_buf_info[root_index].num_rows,
      reinterpret_cast<void*>(h_dst_bufs[partition_index] + h_dst_buf_info[root_index].dst_offset),
      nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[partition_index] +
                                                 h_dst_buf_info[buf_index].dst_offset)
               : nullptr,
      nullable ? UNKNOWN_NULL_COUNT : 0});
    return buf_index + (nullable ? 2 : 1);
  }
};

template <typename InputIter>
size_type build_output_columns(InputIter begin,
                               InputIter end,
                               std::vector<column_view>& dst,
                               int partition_index,
                               int buf_index,
                               uint8_t** h_dst_bufs,
                               dst_buf_info* h_dst_buf_info)
{
  std::for_each(begin, end, [&](column_view const& col) {
    buf_index = cudf::type_dispatcher(col.type(),
                                      build_column_functor{},
                                      col,
                                      dst,
                                      partition_index,
                                      buf_index,
                                      h_dst_bufs,
                                      h_dst_buf_info);
  });
  return buf_index;
}

/**
 * @brief Functor that retrieves the size of a destination buffer
 */
struct buf_size_functor {
  dst_buf_info const* ci;
  size_t operator() __device__(int index) { return static_cast<size_t>(ci[index].buf_size); }
};

/**
 * @brief Functor that retrieves the split "key" for a given output
 * buffer index.
 *
 * The key is simply the partition index.
 */
struct split_key_functor {
  int num_columns;
  int operator() __device__(int buf_index) { return buf_index / num_columns; }
};

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = int;
  using difference_type   = int;
  using pointer           = int*;
  using reference         = int&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i)
  {
    return dst_offset_output_iterator{c + i};
  }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 *
 */
struct size_of_helper {
  template <typename T>
  constexpr std::enable_if_t<not is_fixed_width<T>(), int> __device__ operator()() const
  {
    return 0;
  }

  template <typename T>
  constexpr std::enable_if_t<is_fixed_width<T>(), int> __device__ operator()() const noexcept
  {
    return sizeof(T);
  }
};

};  // anonymous namespace

namespace detail {

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{
  if (input.num_columns() == 0) { return {}; }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns");
  }

  size_t num_root_columns = input.num_columns();

  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers
  size_type num_src_bufs = count_src_bufs(input.begin(), input.end());
  size_t num_partitions  = splits.size() + 1;
  size_t num_bufs        = num_src_bufs * num_partitions;

  // compute total size of host-side temp data
  size_t indices_size =
    cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), split_align);
  size_t src_buf_info_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align);
  size_t buf_sizes_size = cudf::util::round_up_safe(num_partitions * sizeof(size_t), split_align);
  size_t dst_buf_info_size =
    cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align);
  size_t src_bufs_size = cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align);
  size_t dst_bufs_size = cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align);
  size_t total_temp_size = indices_size + src_buf_info_size + buf_sizes_size + dst_buf_info_size +
                           src_bufs_size + dst_bufs_size;

  // clang-format off
  // allocate host
  std::vector<uint8_t> host_buf(total_temp_size);
  // distribute  
  uint8_t* cur_h_buf   = host_buf.data();
  size_type* h_indices = reinterpret_cast<size_type*>(cur_h_buf);             cur_h_buf += indices_size;
  src_buf_info* h_src_buf_info = reinterpret_cast<src_buf_info*>(cur_h_buf);  cur_h_buf += src_buf_info_size;
  size_t* h_buf_sizes = reinterpret_cast<size_t*>(cur_h_buf);                 cur_h_buf += buf_sizes_size;
  dst_buf_info* h_dst_buf_info = reinterpret_cast<dst_buf_info*>(cur_h_buf);  cur_h_buf += dst_buf_info_size;
  uint8_t** h_src_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);              cur_h_buf += src_bufs_size;
  uint8_t** h_dst_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);

  // compute stack space needed for nested list offset calculation (needed on gpu only)
  int offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
  int offset_stack_size           = offset_stack_partition_size * num_partitions * 4;

  // allocate device
  size_t total_device_temp_size = total_temp_size + offset_stack_size;
  rmm::device_buffer device_buf{total_device_temp_size, stream, mr};
  // distribute
  uint8_t* cur_d_buf   = reinterpret_cast<uint8_t*>(device_buf.data());
  size_type* d_indices = reinterpret_cast<size_type*>(cur_d_buf);             cur_d_buf += indices_size;
  src_buf_info* d_src_buf_info = reinterpret_cast<src_buf_info*>(cur_d_buf);  cur_d_buf += src_buf_info_size;
  size_t* d_buf_sizes = reinterpret_cast<size_t*>(cur_d_buf);                 cur_d_buf += buf_sizes_size;
  dst_buf_info* d_dst_buf_info = reinterpret_cast<dst_buf_info*>(cur_d_buf);  cur_d_buf += dst_buf_info_size;
  uint8_t** d_src_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);              cur_d_buf += src_bufs_size;
  uint8_t** d_dst_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);              cur_d_buf += dst_bufs_size;
  size_type* d_offset_stack = reinterpret_cast<size_type*>(cur_d_buf);
  // clang-format on

  // compute splits -> indices.
  {
    size_type* indices = h_indices;
    *indices           = 0;
    indices++;
    std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
      *indices = split;
      indices++;
    });
    *indices = input.column(0).size();

    for (size_t i = 0; i < splits.size(); i++) {
      auto begin = h_indices[i];
      auto end   = h_indices[i + 1];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
    }
  }

  // setup source buf info
  setup_source_buf_info(input.begin(), input.end(), h_src_buf_info, {h_src_buf_info, 0});

  // HtoD indices and source buf info to device
  cudaMemcpyAsync(
    d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_bufs),
    d_dst_buf_info,
    [num_src_bufs,
     d_indices,
     d_src_buf_info,
     d_offset_stack,
     offset_stack_partition_size] __device__(size_t t) {
      int split_index      = t / num_src_bufs;
      int src_buf_index    = t % num_src_bufs;
      auto const& src_info = d_src_buf_info[src_buf_index];

      int row_index_start = d_indices[split_index];
      int row_index_end   = d_indices[split_index + 1];
      int value_shift     = 0;
      int bit_shift       = 0;

      // apply nested offsets (lists and string columns)
      int stack_pos = src_info.offset_stack_pos + (split_index * offset_stack_partition_size);
      size_type* offset_stack  = &d_offset_stack[stack_pos];
      int parent_offsets_index = src_info.parent_offsets_index;
      int stack_size           = 0;

      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      while (stack_size > 0) {
        stack_size--;
        row_index_start = d_src_buf_info[offset_stack[stack_size]].offsets[row_index_start];
        row_index_end   = d_src_buf_info[offset_stack[stack_size]].offsets[row_index_end];
      }

      // if I am an offsets column, all my values need to be shifted
      if (src_info.offsets != nullptr) { value_shift = src_info.offsets[row_index_start]; }
      int num_rows     = row_index_end - row_index_start;
      int num_elements = num_rows;
      if (src_info.offsets != nullptr) { num_elements++; }
      if (src_info.is_validity) {
        bit_shift    = row_index_start % 32;
        num_elements = (num_elements + 31) / 32;
        row_index_start /= 32;
        row_index_end /= 32;
      }
      int element_size = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      size_t bytes     = num_elements * element_size;
      return dst_buf_info{_round_up_safe(bytes, 64),
                          num_elements,
                          element_size,
                          num_rows,
                          row_index_start,
                          0,
                          value_shift,
                          bit_shift};
    });

  // DtoH buf sizes and dest buf info back to the host
  cudaMemcpyAsync(
    h_buf_sizes, d_buf_sizes, buf_sizes_size + dst_buf_info_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // compute total size of each partition
  {
    // key is split index
    auto keys   = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                split_key_functor{static_cast<int>(num_src_bufs)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                  buf_size_functor{d_dst_buf_info});

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          keys,
                          keys + num_bufs,
                          values,
                          thrust::make_discard_iterator(),
                          d_buf_sizes);
  }

  // compute start offset for each output buffer
  {
    auto keys   = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                split_key_functor{static_cast<int>(num_src_bufs)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                  buf_size_functor{d_dst_buf_info});
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                  keys,
                                  keys + num_bufs,
                                  values,
                                  dst_offset_output_iterator{d_dst_buf_info},
                                  0);
  }

  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(
    h_buf_sizes, d_buf_sizes, buf_sizes_size + dst_buf_info_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // allocate output partition buffers
  std::vector<rmm::device_buffer> out_buffers;
  out_buffers.reserve(num_partitions);
  std::transform(h_buf_sizes,
                 h_buf_sizes + num_partitions,
                 std::back_inserter(out_buffers),
                 [stream, mr](size_t bytes) {
                   return rmm::device_buffer{bytes, stream, mr};
                 });

  // setup src buffers
  setup_src_buf_data(input.begin(), input.end(), 0, h_src_bufs);

  // setup dst buffers
  {
    size_type out_index = 0;
    std::for_each(
      out_buffers.begin(), out_buffers.end(), [&out_index, &h_dst_bufs](rmm::device_buffer& buf) {
        h_dst_bufs[out_index++] = reinterpret_cast<uint8_t*>(buf.data());
      });
  }

  // HtoD src and dest buffers
  cudaMemcpyAsync(
    d_src_bufs, h_src_bufs, src_bufs_size + dst_bufs_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // copy.  1 block per buffer
  {
    // scope_timer timer("kernel");
    constexpr int block_size = 512;
    copy_partition<<<num_bufs, block_size, 0, stream>>>(
      num_src_bufs, num_partitions, d_src_bufs, d_dst_bufs, d_dst_buf_info);

    // TODO : put this after the column building step below to overlap work
    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // build the output.
  std::vector<contiguous_split_result> result;
  result.reserve(num_partitions);
  std::vector<column_view> cols;
  cols.reserve(num_root_columns);
  size_t buf_index = 0;
  for (size_t idx = 0; idx < num_partitions; idx++) {
    buf_index = build_output_columns(
      input.begin(), input.end(), cols, idx, buf_index, h_dst_bufs, h_dst_buf_info);
    result.push_back(contiguous_split_result{
      cudf::table_view{cols}, std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))});
    cols.clear();
  }

  return std::move(result);
}

};  // namespace detail

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::contiguous_split(input, splits, mr, (cudaStream_t)0);
}

};  // namespace cudf