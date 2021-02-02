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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>

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
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a seperate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               const int* _offsets,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               int _column_offset)
    : type(_type),
      offsets(_offsets),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  cudf::type_id type;
  const int* offsets;        // a pointer to device memory offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
  size_type column_offset;   // offset in the case of a sliced column
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
 * @param dst Destination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param num_elements Number of elements to copy
 * @param element_size Size of each element in bytes
 * @param src_row_index Row index to start copying at
 * @param stride Size of the kernel block
 * @param value_shift Shift incoming 4-byte offset values down by this amount
 * @param bit_shift Shift incoming data right by this many bits
 */
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t* __restrict__ src,
                            int t,
                            int num_elements,
                            int element_size,
                            int src_row_index,
                            uint32_t stride,
                            int value_shift,
                            int bit_shift)
{
  src += (src_row_index * element_size);

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
 * When doing a contiguous_split on X columns comprising N total internal buffers
 * with M splits, we end up having to copy N*M source/destination buffer pairs.
 * This kernel is arranged such that each block copies 1 source/destination pair.
 * This function retrieves the relevant buffers and then calls copy_buffer to perform
 * the actual copy.
 *
 * @param num_src_bufs Total number of source buffers (N)
 * @param num_partitions Number of partitions the each source buffer is split into (M)
 * @param src_bufs Input source buffers (N)
 * @param dst_bufs Desination buffers (N*M)
 * @param buf_info Information on the range of values to be copied for each destination buffer.
 */
__global__ void copy_partition(int num_src_bufs,
                               int num_partitions,
                               uint8_t** src_bufs,
                               uint8_t** dst_bufs,
                               dst_buf_info* buf_info)
{
  int const partition_index = blockIdx.x / num_src_bufs;
  int const src_buf_index   = blockIdx.x % num_src_bufs;
  size_t const buf_index    = (partition_index * num_src_bufs) + src_buf_index;

  // copy, shifting offsets and validity bits as needed
  copy_buffer(dst_bufs[partition_index] + buf_info[buf_index].dst_offset,
              src_bufs[src_buf_index],
              threadIdx.x,
              buf_info[buf_index].num_elements,
              buf_info[buf_index].element_size,
              buf_info[buf_index].src_row_index,
              blockDim.x,
              buf_info[buf_index].value_shift,
              buf_info[buf_index].bit_shift);
}

// The block of functions below are all related:
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
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

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
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns.
 */
template <typename InputIter>
size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (col.nullable() ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
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
 * @param out_buf Iterator into output buffer infos
 *
 * @returns next output buffer iterator
 */
template <typename InputIter, typename OutputIter>
OutputIter setup_src_buf_data(InputIter begin, InputIter end, OutputIter out_buf)
{
  std::for_each(begin, end, [&out_buf](column_view const& col) {
    if (col.nullable()) {
      *out_buf = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
      out_buf++;
    }
    // NOTE: we're always returning the base pointer here.  column-level offset is accounted
    // for later. Also, for some column types (string, list, struct) this pointer will be null
    // because there is no associated data with the root column.
    *out_buf = const_cast<uint8_t*>(col.head<uint8_t>());
    out_buf++;

    out_buf = setup_src_buf_data(col.child_begin(), col.child_end(), out_buf);
  });
  return out_buf;
}

/**
 * @brief Count the total number of source buffers we will be copying
 * from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (data, validity).
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
size_type count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    return 1 + (col.nullable() ? 1 : 0) + count_src_bufs(col.child_begin(), col.child_end());
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), 0);
}

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
 * @param current Current source buffer info to be written to
 * @param offset_stack_pos Integer representing our current offset nesting depth
 * (how many list or string levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          int offset_stack_pos    = 0,
                                                          int parent_offset_index = -1,
                                                          int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  std::pair<src_buf_info*, size_type> operator()(column_view const& col,
                                                 src_buf_info* current,
                                                 int offset_stack_pos,
                                                 int parent_offset_index,
                                                 int offset_depth)
  {
    if (col.nullable()) {
      std::tie(current, offset_stack_pos) =
        add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
    }

    // info for the data buffer
    *current = src_buf_info(
      col.type().id(), nullptr, offset_stack_pos, parent_offset_index, false, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }

 private:
  std::pair<src_buf_info*, size_type> add_null_buffer(column_view const& col,
                                                      src_buf_info* current,
                                                      int offset_stack_pos,
                                                      int parent_offset_index,
                                                      int offset_depth)
  {
    // info for the validity buffer
    *current = src_buf_info(
      type_id::INT32, nullptr, offset_stack_pos, parent_offset_index, true, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }
};

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::string_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // string columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::STRING, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // string columns don't necessarily have children
  if (col.num_children() > 0) {
    CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    auto offset_col = current;
    CUDF_EXPECTS(scv.offsets().nullable() == false, "Encountered nullable string offsets column");
    *current = src_buf_info(type_id::INT32,
                            // note: offsets can be null in the case where the string column
                            // has been created with empty_like().
                            scv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                            offset_stack_pos,
                            parent_offset_index,
                            false,
                            col.offset());

    current++;
    offset_stack_pos += offset_depth;

    // since we are crossing an offset boundary, calculate our new depth and parent offset index.
    offset_depth++;
    parent_offset_index = offset_col - head;

    // prevent appending buf_info for non-existent chars buffer
    CUDF_EXPECTS(scv.chars().nullable() == false, "Encountered nullable string chars column");

    // info for the chars buffer
    *current = src_buf_info(
      type_id::INT8, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
    current++;
    offset_stack_pos += offset_depth;
  }

  return {current, offset_stack_pos};
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::list_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  lists_column_view lcv(col);

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::LIST, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  auto offset_col = current;
  *current        = src_buf_info(type_id::INT32,
                          // note: offsets can be null in the case where the lists column
                          // has been created with empty_like().
                          lcv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                          offset_stack_pos,
                          parent_offset_index,
                          false,
                          col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // since we are crossing an offset boundary, calculate our new depth and parent offset index.
  offset_depth++;
  parent_offset_index = offset_col - head;

  return setup_source_buf_info(col.child_begin() + 1,
                               col.child_end(),
                               head,
                               current,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::struct_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::STRUCT, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // recurse on children
  return setup_source_buf_info(col.child_begin(),
                               col.child_end(),
                               head,
                               current,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::dictionary32>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  CUDF_FAIL("Unsupported type");
}

template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          int offset_stack_pos,
                                                          int parent_offset_index,
                                                          int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    std::tie(current, offset_stack_pos) = cudf::type_dispatcher(col.type(),
                                                                buf_info_functor{head},
                                                                col,
                                                                current,
                                                                offset_stack_pos,
                                                                parent_offset_index,
                                                                offset_depth);
  });
  return {current, offset_stack_pos};
}

/**
 * @brief Given a set of input columns and processed split buffers, produce
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
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param out_begin Output iterator of column views
 * @param base_ptr Pointer to the base address of copied data for the working partition
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
template <typename InputIter, typename BufInfo, typename Output>
BufInfo build_output_columns(InputIter begin,
                             InputIter end,
                             BufInfo info_begin,
                             Output out_begin,
                             uint8_t const* const base_ptr)
{
  auto current_info = info_begin;
  std::transform(begin, end, out_begin, [&current_info, base_ptr](column_view const& src) {
    // Use C++17 structured bindings
    bitmask_type const* bitmask_ptr;
    size_type null_count;
    std::tie(bitmask_ptr, null_count) = [&]() {
      if (src.nullable()) {
        auto const ptr = reinterpret_cast<bitmask_type const*>(base_ptr + current_info->dst_offset);
        ++current_info;
        return std::make_pair(ptr, UNKNOWN_NULL_COUNT);
      }
      return std::make_pair(static_cast<bitmask_type const*>(nullptr), 0);
    }();

    // size/data pointer for the column
    auto const size = current_info->num_elements;
    uint8_t const* data_ptr =
      size == 0 || src.head() == nullptr ? nullptr : base_ptr + current_info->dst_offset;
    ++current_info;

    // children
    auto children = std::vector<column_view>{};
    children.reserve(src.num_children());
    current_info = build_output_columns(
      src.child_begin(), src.child_end(), current_info, std::back_inserter(children), base_ptr);

    return column_view{src.type(), size, data_ptr, bitmask_ptr, null_count, 0, std::move(children)};
  });

  return current_info;
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
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

};  // anonymous namespace

namespace detail {

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0) { return {}; }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns");
  }
  {
    size_type begin = 0;
    for (size_t i = 0; i < splits.size(); i++) {
      size_type end = splits[i];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
      begin = end;
    }
  }

  size_t const num_partitions = splits.size() + 1;

  // if inputs are empty, just return num_partitions empty tables
  if (input.column(0).size() == 0) {
    std::vector<contiguous_split_result> result;
    result.reserve(num_partitions);

    auto iter = thrust::make_counting_iterator(0);
    std::transform(
      iter, iter + num_partitions, std::back_inserter(result), [&input](int partition_index) {
        return contiguous_split_result{input, std::make_unique<rmm::device_buffer>()};
      });

    return result;
  }

  size_t const num_root_columns = input.num_columns();

  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers
  size_type const num_src_bufs = count_src_bufs(input.begin(), input.end());
  size_t const num_bufs        = num_src_bufs * num_partitions;

  // packed block of memory 1. split indices and src_buf_info structs
  size_t const indices_size =
    cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), split_align);
  size_t const src_buf_info_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align);
  // host-side
  std::vector<uint8_t> h_indices_and_source_info(indices_size + src_buf_info_size);
  size_type* h_indices = reinterpret_cast<size_type*>(h_indices_and_source_info.data());
  src_buf_info* h_src_buf_info =
    reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size);
  // device-side
  // gpu-only : stack space needed for nested list offset calculation
  int const offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
  size_t const offset_stack_size = offset_stack_partition_size * num_partitions * sizeof(size_type);
  rmm::device_buffer d_indices_and_source_info(indices_size + src_buf_info_size + offset_stack_size,
                                               stream,
                                               rmm::mr::get_current_device_resource());
  size_type* d_indices         = reinterpret_cast<size_type*>(d_indices_and_source_info.data());
  src_buf_info* d_src_buf_info = reinterpret_cast<src_buf_info*>(
    reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) + indices_size);
  size_type* d_offset_stack =
    reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) +
                                 indices_size + src_buf_info_size);

  // compute splits -> indices.
  h_indices[0]              = 0;
  h_indices[num_partitions] = input.column(0).size();
  std::copy(splits.begin(), splits.end(), std::next(h_indices));

  // setup source buf info
  setup_source_buf_info(input.begin(), input.end(), h_src_buf_info, h_src_buf_info);

  // HtoD indices and source buf info to device
  CUDA_TRY(cudaMemcpyAsync(d_indices,
                           h_indices,
                           indices_size + src_buf_info_size,
                           cudaMemcpyHostToDevice,
                           stream.value()));

  // packed block of memory 2. partition buffer sizes and dst_buf_info structs
  size_t const buf_sizes_size =
    cudf::util::round_up_safe(num_partitions * sizeof(size_t), split_align);
  size_t const dst_buf_info_size =
    cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align);
  // host-side
  std::vector<uint8_t> h_buf_sizes_and_dst_info(buf_sizes_size + dst_buf_info_size);
  size_t* h_buf_sizes = reinterpret_cast<size_t*>(h_buf_sizes_and_dst_info.data());
  dst_buf_info* h_dst_buf_info =
    reinterpret_cast<dst_buf_info*>(h_buf_sizes_and_dst_info.data() + buf_sizes_size);
  // device-side
  rmm::device_buffer d_buf_sizes_and_dst_info(
    buf_sizes_size + dst_buf_info_size, stream, rmm::mr::get_current_device_resource());
  size_t* d_buf_sizes          = reinterpret_cast<size_t*>(d_buf_sizes_and_dst_info.data());
  dst_buf_info* d_dst_buf_info = reinterpret_cast<dst_buf_info*>(
    static_cast<uint8_t*>(d_buf_sizes_and_dst_info.data()) + buf_sizes_size);

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_bufs),
    d_dst_buf_info,
    [num_src_bufs,
     d_indices,
     d_src_buf_info,
     d_offset_stack,
     offset_stack_partition_size] __device__(size_t t) {
      int const split_index   = t / num_src_bufs;
      int const src_buf_index = t % num_src_bufs;
      auto const& src_info    = d_src_buf_info[src_buf_index];

      // apply nested offsets (lists and string columns).
      //
      // We can't just use the incoming row indices to figure out where to read from in a
      // nested list situation.  We have to apply offsets every time we cross a boundary
      // (list or string).  This loop applies those offsets so that our incoming row_index_start
      // and row_index_end get transformed to our final values.
      //
      int const stack_pos = src_info.offset_stack_pos + (split_index * offset_stack_partition_size);
      size_type* offset_stack  = &d_offset_stack[stack_pos];
      int parent_offsets_index = src_info.parent_offsets_index;
      int stack_size           = 0;
      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      // make sure to include the -column- offset in our calculations
      int row_start = d_indices[split_index] + src_info.column_offset;
      int row_end   = d_indices[split_index + 1] + src_info.column_offset;
      while (stack_size > 0) {
        stack_size--;
        auto const offsets = d_src_buf_info[offset_stack[stack_size]].offsets;
        // this case can happen when you have empty string or list columns constructed with
        // empty_like()
        if (offsets != nullptr) {
          row_start = offsets[row_start];
          row_end   = offsets[row_end];
        }
      }

      // final row indices and row count
      int const out_row_index = src_info.is_validity ? row_start / 32 : row_start;
      int const num_rows      = row_end - row_start;
      // if I am an offsets column, all my values need to be shifted
      int const value_shift = src_info.offsets == nullptr ? 0 : src_info.offsets[row_start];
      // if I am a validity column, we may need to shift bits
      int const bit_shift = src_info.is_validity ? row_start % 32 : 0;
      // # of rows isn't necessarily the same as # of elements to be copied.
      auto const num_elements = [&]() {
        if (src_info.offsets != nullptr && num_rows > 0) {
          return num_rows + 1;
        } else if (src_info.is_validity) {
          return (num_rows + 31) / 32;
        }
        return num_rows;
      }();
      int const element_size = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      size_t const bytes     = num_elements * element_size;
      return dst_buf_info{_round_up_safe(bytes, 64),
                          num_elements,
                          element_size,
                          num_rows,
                          out_row_index,
                          0,
                          value_shift,
                          bit_shift};
    });

  // compute total size of each partition
  {
    // key is split index
    auto keys   = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                split_key_functor{static_cast<int>(num_src_bufs)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                  buf_size_functor{d_dst_buf_info});

    thrust::reduce_by_key(rmm::exec_policy(stream),
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
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                  keys,
                                  keys + num_bufs,
                                  values,
                                  dst_offset_output_iterator{d_dst_buf_info},
                                  0);
  }

  // DtoH buf sizes and col info back to the host
  CUDA_TRY(cudaMemcpyAsync(h_buf_sizes,
                           d_buf_sizes,
                           buf_sizes_size + dst_buf_info_size,
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  stream.synchronize();

  // allocate output partition buffers
  std::vector<rmm::device_buffer> out_buffers;
  out_buffers.reserve(num_partitions);
  std::transform(h_buf_sizes,
                 h_buf_sizes + num_partitions,
                 std::back_inserter(out_buffers),
                 [stream, mr](size_t bytes) {
                   return rmm::device_buffer{bytes, stream, mr};
                 });

  // packed block of memory 3. pointers to source and destination buffers (and stack space on the
  // gpu for offset computation)
  size_t const src_bufs_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align);
  size_t const dst_bufs_size =
    cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align);
  // host-side
  std::vector<uint8_t> h_src_and_dst_buffers(src_bufs_size + dst_bufs_size);
  uint8_t** h_src_bufs = reinterpret_cast<uint8_t**>(h_src_and_dst_buffers.data());
  uint8_t** h_dst_bufs = reinterpret_cast<uint8_t**>(h_src_and_dst_buffers.data() + src_bufs_size);
  // device-side
  rmm::device_buffer d_src_and_dst_buffers(src_bufs_size + dst_bufs_size + offset_stack_size,
                                           stream,
                                           rmm::mr::get_current_device_resource());
  uint8_t** d_src_bufs = reinterpret_cast<uint8_t**>(d_src_and_dst_buffers.data());
  uint8_t** d_dst_bufs = reinterpret_cast<uint8_t**>(
    reinterpret_cast<uint8_t*>(d_src_and_dst_buffers.data()) + src_bufs_size);

  // setup src buffers
  setup_src_buf_data(input.begin(), input.end(), h_src_bufs);

  // setup dst buffers
  std::transform(out_buffers.begin(), out_buffers.end(), h_dst_bufs, [](auto& buf) {
    return static_cast<uint8_t*>(buf.data());
  });

  // HtoD src and dest buffers
  CUDA_TRY(cudaMemcpyAsync(
    d_src_bufs, h_src_bufs, src_bufs_size + dst_bufs_size, cudaMemcpyHostToDevice, stream.value()));

  // copy.  1 block per buffer
  {
    constexpr int block_size = 512;
    copy_partition<<<num_bufs, block_size, 0, stream.value()>>>(
      num_src_bufs, num_partitions, d_src_bufs, d_dst_bufs, d_dst_buf_info);
  }

  // build the output.
  std::vector<contiguous_split_result> result;
  result.reserve(num_partitions);
  std::vector<column_view> cols;
  cols.reserve(num_root_columns);
  auto cur_dst_buf_info = h_dst_buf_info;
  for (size_t idx = 0; idx < num_partitions; idx++) {
    cur_dst_buf_info = build_output_columns(
      input.begin(), input.end(), cur_dst_buf_info, std::back_inserter(cols), h_dst_bufs[idx]);
    result.push_back(contiguous_split_result{
      cudf::table_view{cols}, std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))});
    cols.clear();
  }

  // we need to synchronize, as the HtoD memcpy before the copy kernel could still be in-flight as
  // we exit the function and our stack frame disappears. also of note : we're overlapping
  // construction of the output columns on the cpu while the data they point to is still being
  // copied. this can be a significant time savings.
  stream.synchronize();

  return result;
}

};  // namespace detail

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::contiguous_split(input, splits, rmm::cuda_stream_default, mr);
}

};  // namespace cudf
