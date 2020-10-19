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
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/discard_iterator.h>

#include <numeric>

#define __NEW_PATH

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Copies contents of `in` to `out`.  Copies validity if present
 * but does not compute null count.
 *
 * @param in column_view to copy from
 * @param out mutable_column_view to copy to.
 */
template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size) __global__
  void copy_in_place_kernel(column_device_view const in, mutable_column_device_view out)
{
  const size_type tid            = threadIdx.x + blockIdx.x * block_size;
  const int warp_id              = tid / cudf::detail::warp_size;
  const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;

  // begin/end indices for the column data
  size_type begin = 0;
  size_type end   = in.size();
  // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
  // each warp will process one (32 bit) of the validity mask via
  // __ballot_sync()
  size_type warp_begin = cudf::word_index(begin);
  size_type warp_end   = cudf::word_index(end - 1);

  // lane id within the current warp
  const int lane_id = threadIdx.x % cudf::detail::warp_size;

  // current warp.
  size_type warp_cur = warp_begin + warp_id;
  size_type index    = tid;
  while (warp_cur <= warp_end) {
    bool in_range = (index >= begin && index < end);

    bool valid = true;
    if (has_validity) { valid = in_range && in.is_valid(index); }
    if (in_range) { out.element<T>(index) = in.element<T>(index); }

    // update validity
    if (has_validity) {
      // the final validity mask for this warp
      int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
      // only one guy in the warp needs to update the mask and count
      if (lane_id == 0) { out.set_mask_word(warp_cur, warp_mask); }
    }

    // next grid
    warp_cur += warps_per_grid;
    index += block_size * gridDim.x;
  }
}

/**
 * @brief Copies contents of one string column to another.  Copies validity if present
 * but does not compute null count.
 *
 * The purpose of this kernel is to reduce the number of
 * kernel calls for copying a string column from 2 to 1, since number of kernel calls is the
 * dominant factor in large scale contiguous_split() calls.  To do this, the kernel is
 * invoked with using max(num_chars, num_offsets) threads and then doing separate
 * bounds checking on offset, chars and validity indices.
 *
 * Outgoing offset values are shifted down to account for the new base address
 * each column gets as a result of the contiguous_split() process.
 *
 * @param in num_strings number of strings (rows) in the column
 * @param in offsets_in pointer to incoming offsets to be copied
 * @param out offsets_out pointer to output offsets
 * @param in validity_in_offset offset into validity buffer to add to element indices
 * @param in validity_in pointer to incoming validity vector to be copied
 * @param out validity_out pointer to output validity vector
 * @param in offset_shift value to shift copied offsets down by
 * @param in num_chars number of chars to copy
 * @param in chars_in input chars to be copied
 * @param out chars_out output chars to be copied.
 */
template <size_type block_size, bool has_validity>
__launch_bounds__(block_size) __global__
  void copy_in_place_strings_kernel(size_type num_strings,
                                    size_type const* __restrict__ offsets_in,
                                    size_type* __restrict__ offsets_out,
                                    size_type validity_in_offset,
                                    bitmask_type const* __restrict__ validity_in,
                                    bitmask_type* __restrict__ validity_out,
                                    size_type offset_shift,
                                    size_type num_chars,
                                    char const* __restrict__ chars_in,
                                    char* __restrict__ chars_out)
{
  const size_type tid            = threadIdx.x + blockIdx.x * block_size;
  const int warp_id              = tid / cudf::detail::warp_size;
  const size_type warps_per_grid = gridDim.x * block_size / cudf::detail::warp_size;

  // how many warps we'll be processing. with strings, the chars and offsets
  // lengths may be different.  so we'll just march the worst case.
  size_type warp_begin = cudf::word_index(0);
  size_type warp_end   = cudf::word_index(std::max(num_chars, num_strings + 1) - 1);

  // end indices for chars
  size_type chars_end = num_chars;
  // end indices for offsets
  size_type offsets_end = num_strings + 1;
  // end indices for validity and the last warp that actually should
  // be updated
  size_type validity_end      = num_strings;
  size_type validity_warp_end = cudf::word_index(num_strings - 1);

  // lane id within the current warp
  const int lane_id = threadIdx.x % cudf::detail::warp_size;

  size_type warp_cur = warp_begin + warp_id;
  size_type index    = tid;
  while (warp_cur <= warp_end) { 
    if (index < chars_end) { chars_out[index] = chars_in[index]; }

    if (index < offsets_end) {
      // each output column starts at a new base pointer. so we have to
      // shift every offset down by the point (in chars) at which it was split.
      offsets_out[index] = offsets_in[index] - offset_shift;
    }

    // if we're still in range of validity at all
    if (has_validity && warp_cur <= validity_warp_end) {
      bool valid = (index < validity_end) && bit_is_set(validity_in, validity_in_offset + index);

      // the final validity mask for this warp
      int warp_mask = __ballot_sync(0xFFFF'FFFF, valid);
      // only one guy in the warp needs to update the mask and count
      if (lane_id == 0) { validity_out[warp_cur] = warp_mask; }
    }

    // next grid
    warp_cur += warps_per_grid;
    index += block_size * gridDim.x;
  }
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;

/**
 * @brief Information about the split for a given column. Bundled together
 *        into a struct because tuples were getting pretty unreadable.
 */
struct column_split_info {
  size_t data_buf_size;      // size of the data (including padding)
  size_t validity_buf_size;  // validity vector size (including padding)

  size_t offsets_buf_size;  // (strings only) size of offset column (including padding)
  size_type num_chars;      // (strings only) number of chars in the column
  size_type chars_offset;   // (strings only) offset from head of chars data
};

/**
 * @brief Functor called by the `type_dispatcher` to incrementally compute total
 * memory buffer size needed to allocate a contiguous copy of all columns within
 * a source table.
 */
struct column_buffer_size_functor {
  template <typename T>
  size_t operator()(column_view const& c, column_split_info& split_info)
  {
    split_info.data_buf_size = cudf::util::round_up_safe(c.size() * sizeof(T), split_align);
    split_info.validity_buf_size =
      (c.has_nulls() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
    return split_info.data_buf_size + split_info.validity_buf_size;
  }
};
template <>
size_t column_buffer_size_functor::operator()<string_view>(column_view const& c,
                                                           column_split_info& split_info)
{
  // this has already been precomputed in an earlier step. return the sum.
  return split_info.data_buf_size + split_info.validity_buf_size + split_info.offsets_buf_size;
}

/**
 * @brief Functor called by the `type_dispatcher` to copy a column into a contiguous
 * buffer of output memory.
 *
 * Used for copying each column in a source table into one contiguous buffer of memory.
 */
struct column_copy_functor {
  template <typename T>
  void operator()(column_view const& in,
                  column_split_info const& split_info,
                  char*& dst,
                  std::vector<column_view>& out_cols)
  {
    // outgoing pointers
    char* data             = dst;
    bitmask_type* validity = split_info.validity_buf_size == 0
                               ? nullptr
                               : reinterpret_cast<bitmask_type*>(dst + split_info.data_buf_size);

    // increment working buffer
    dst += (split_info.data_buf_size + split_info.validity_buf_size);

    // no work to do
    if (in.size() == 0) {
      out_cols.push_back(column_view{in.type(), 0, nullptr});
      return;
    }

    // custom copy kernel (which could probably just be an in-place copy() function in cudf).
    cudf::size_type num_els  = cudf::util::round_up_safe(in.size(), cudf::detail::warp_size);
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{num_els, block_size, 1};

    // output copied column
    mutable_column_view mcv{in.type(), in.size(), data, validity, in.null_count()};
    if (in.has_nulls()) {
      copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
        *column_device_view::create(in), *mutable_column_device_view::create(mcv));
    } else {
      copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
        *column_device_view::create(in), *mutable_column_device_view::create(mcv));
    }

    out_cols.push_back(mcv);
  }
};
template <>
void column_copy_functor::operator()<string_view>(column_view const& in,
                                                  column_split_info const& split_info,
                                                  char*& dst,
                                                  std::vector<column_view>& out_cols)
{
  // outgoing pointers
  char* chars_buf            = dst;
  bitmask_type* validity_buf = split_info.validity_buf_size == 0
                                 ? nullptr
                                 : reinterpret_cast<bitmask_type*>(dst + split_info.data_buf_size);
  size_type* offsets_buf =
    reinterpret_cast<size_type*>(dst + split_info.data_buf_size + split_info.validity_buf_size);

  // increment working buffer
  dst += (split_info.data_buf_size + split_info.validity_buf_size + split_info.offsets_buf_size);

  // offsets column.
  strings_column_view strings_c(in);
  column_view in_offsets = strings_c.offsets();
  // note, incoming columns are sliced, so their size is fundamentally different from their child
  // offset columns, which are unsliced.
  size_type num_offsets = in.size() + 1;
  cudf::size_type num_threads =
    cudf::util::round_up_safe(std::max(split_info.num_chars, num_offsets), cudf::detail::warp_size);
  column_view in_chars = strings_c.chars();

  // a column with no strings will still have a single offset.
  CUDF_EXPECTS(num_offsets > 0, "Invalid offsets child column");

  // 1 combined kernel call that copies chars, offsets and validity in one pass. see notes on why
  // this exists in the kernel brief.
  constexpr int block_size = 256;
  cudf::detail::grid_1d grid{num_threads, block_size, 1};
  if (in.has_nulls()) {
    copy_in_place_strings_kernel<block_size, true><<<grid.num_blocks, block_size, 0, 0>>>(
      in.size(),                                        // num_rows
      in_offsets.head<size_type>() + in.offset(),       // offsets_in
      offsets_buf,                                      // offsets_out
      in.offset(),                                      // validity_in_offset
      in.null_mask(),                                   // validity_in
      validity_buf,                                     // validity_out
      split_info.chars_offset,                          // offset_shift
      split_info.num_chars,                             // num_chars
      in_chars.head<char>() + split_info.chars_offset,  // chars_in
      chars_buf);
  } else {
    copy_in_place_strings_kernel<block_size, false><<<grid.num_blocks, block_size, 0, 0>>>(
      in.size(),                                        // num_rows
      in_offsets.head<size_type>() + in.offset(),       // offsets_in
      offsets_buf,                                      // offsets_out
      0,                                                // validity_in_offset
      nullptr,                                          // validity_in
      nullptr,                                          // validity_out
      split_info.chars_offset,                          // offset_shift
      split_info.num_chars,                             // num_chars
      in_chars.head<char>() + split_info.chars_offset,  // chars_in
      chars_buf);
  }

  // output child columns
  column_view out_offsets{in_offsets.type(), num_offsets, offsets_buf};
  column_view out_chars{in_chars.type(), static_cast<size_type>(split_info.num_chars), chars_buf};

  // result
  out_cols.push_back(column_view(
    in.type(), in.size(), nullptr, validity_buf, in.null_count(), 0, {out_offsets, out_chars}));
}

/**
 * @brief Information about a string column in a table view.
 *
 * Used internally by preprocess_string_column_info as part of a device-accessible
 * vector for computing final string information in a single kernel call.
 */
struct column_preprocess_info {
  size_type index;
  size_type offset;
  size_type size;
  bool has_nulls;
  cudf::column_device_view offsets;
};

/**
 * @brief Preprocess information about all strings columns in a table view.
 *
 * In order to minimize how often we touch the gpu, we need to preprocess various pieces of
 * information about the string columns in a table as a batch process.  This function builds a list
 * of the offset columns for all input string columns and computes this information with a single
 * thrust call.  In addition, the vector returned is allocated for -all- columns in the table so
 * further processing of non-string columns can happen afterwards.
 *
 * The key things this function avoids
 * - avoiding reaching into gpu memory on the cpu to retrieve offsets to compute string sizes.
 * - creating column_device_views on the base string_column_view itself as that causes gpu memory
 * allocation.
 */
thrust::host_vector<column_split_info> preprocess_string_column_info(
  cudf::table_view const& t,
  rmm::device_vector<column_split_info>& device_split_info,
  cudaStream_t stream)
{
  // build a list of all the offset columns and their indices for all input string columns and put
  // them on the gpu
  thrust::host_vector<column_preprocess_info> offset_columns;
  offset_columns.reserve(t.num_columns());  // worst case

  // collect only string columns
  size_type column_index = 0;
  std::for_each(t.begin(), t.end(), [&offset_columns, &column_index](cudf::column_view const& c) {
    if (c.type().id() == type_id::STRING) {
      cudf::column_device_view cdv((strings_column_view(c)).offsets(), 0, 0);
      offset_columns.push_back(
        column_preprocess_info{column_index, c.offset(), c.size(), c.has_nulls(), cdv});
    }
    column_index++;
  });
  rmm::device_vector<column_preprocess_info> device_offset_columns = offset_columns;

  // compute column split information
  rmm::device_vector<thrust::pair<size_type, size_type>> device_offsets(t.num_columns());
  auto* offsets_p = device_offsets.data().get();
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   device_offset_columns.begin(),
                   device_offset_columns.end(),
                   [offsets_p] __device__(column_preprocess_info const& cpi) {
                     offsets_p[cpi.index] =
                       thrust::make_pair(cpi.offsets.head<int32_t>()[cpi.offset],
                                         cpi.offsets.head<int32_t>()[cpi.offset + cpi.size]);
                   });
  thrust::host_vector<thrust::pair<size_type, size_type>> host_offsets(device_offsets);
  thrust::host_vector<column_split_info> split_info(t.num_columns());
  std::for_each(offset_columns.begin(),
                offset_columns.end(),
                [&split_info, &host_offsets](column_preprocess_info const& cpi) {
                  int32_t offset_start = host_offsets[cpi.index].first;
                  int32_t offset_end   = host_offsets[cpi.index].second;
                  auto num_chars       = offset_end - offset_start;
                  split_info[cpi.index].data_buf_size =
                    cudf::util::round_up_safe(static_cast<size_t>(num_chars), split_align);
                  split_info[cpi.index].validity_buf_size =
                    cpi.has_nulls ? cudf::bitmask_allocation_size_bytes(cpi.size, split_align) : 0;
                  split_info[cpi.index].offsets_buf_size =
                    cudf::util::round_up_safe((cpi.size + 1) * sizeof(size_type), split_align);
                  split_info[cpi.index].num_chars    = num_chars;
                  split_info[cpi.index].chars_offset = offset_start;
                });
  return split_info;
}

/**
 * @brief Creates a contiguous_split_result object which contains a deep-copy of the input
 * table_view into a single contiguous block of memory.
 *
 * The table_view contained within the contiguous_split_result will pass an expect_tables_equal()
 * call with the input table.  The memory referenced by the table_view and its internal column_views
 * is entirely contained in single block of memory.
 */
contiguous_split_result alloc_and_copy(cudf::table_view const& t,
                                       rmm::device_vector<column_split_info>& device_split_info,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream)
{
  // preprocess column split information for string columns.
  thrust::host_vector<column_split_info> split_info =
    preprocess_string_column_info(t, device_split_info, stream);

  // compute the rest of the column sizes (non-string columns, and total buffer size)
  size_t total_size      = 0;
  size_type column_index = 0;
  std::for_each(
    t.begin(), t.end(), [&total_size, &column_index, &split_info](cudf::column_view const& c) {
      total_size +=
        cudf::type_dispatcher(c.type(), column_buffer_size_functor{}, c, split_info[column_index]);
      column_index++;
    });

  // allocate
  auto device_buf = std::make_unique<rmm::device_buffer>(total_size, stream, mr);
  char* buf       = static_cast<char*>(device_buf->data());

  // copy (this would be cleaner with a std::transform, but there's an nvcc compiler issue in the
  // way)
  std::vector<column_view> out_cols;
  out_cols.reserve(t.num_columns());

  column_index = 0;
  std::for_each(
    t.begin(), t.end(), [&out_cols, &buf, &column_index, &split_info](cudf::column_view const& c) {
      cudf::type_dispatcher(
        c.type(), column_copy_functor{}, c, split_info[column_index], buf, out_cols);
      column_index++;
    });

  return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};
}

};  // anonymous namespace

struct _size_of_helper {
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


inline __device__ size_t _round_up_safe(size_t number_to_round, size_t modulus)
{
  auto remainder = number_to_round % modulus;
  if (remainder == 0) { return number_to_round; }
  auto rounded_up = number_to_round - remainder + modulus;
  return rounded_up;
}

struct _src_column_info {
  cudf::type_id     type;
  const int         *offsets;
  bool              is_offset_column;
  bool              is_validity;
};

struct _column_info {
  size_t    buf_size;       // total size of buffer, including padding
  int       num_elements;   // # of elements to be copied
  int       element_size;
  int       src_row_index;
  int       dst_offset;   
  int       value_shift;
  int       bit_shift;  
};

struct dst_offset_output_iterator {
  _column_info *c;
  using value_type        = int;
  using difference_type   = int;
  using pointer           = int *;
  using reference         = int &;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i)
  {
    return dst_offset_output_iterator{c + i};
  }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator*__device__() { return dereference(c); }

 private:
  reference __device__ dereference(_column_info *c)
  {
    return c->dst_offset;
  }
};

__device__ void copy_buffer(uint8_t * __restrict__ dst, uint8_t *__restrict__ _src,
                            int t, int num_elements, int element_size,
                            int src_row_index, uint32_t stride, int value_shift, int bit_shift)
{
  uint8_t *src = _src + (src_row_index * element_size);

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  const size_t num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  const uint32_t ofs = reinterpret_cast<uintptr_t>(src) % 4;
  size_t pos = t*16;
  stride *= 16;
  while(pos+20 <= num_bytes){
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t *>((src + pos) - ofs);
    uint4 v = { in32[0], in32[1], in32[2], in32[3] };
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs*8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs*8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs*8 + bit_shift);
      // don't read off the end
      v.w = __funnelshift_r(v.w, /*pos+20 <= num_bytes ? in32[4] : 0*/in32[4], ofs*8 + bit_shift);
    }           
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos/16] = v;
    pos += stride;
  }  

  // copy trailing bytes
  if(t == 0){
    size_t remainder = num_bytes < 16 ? num_bytes : 16 + (num_bytes % 16);

    /*
    if(blockIdx.x == 80){
      printf("R : num_bytes(%lu), pos(%lu), remainder(%lu)\n", num_bytes, pos, remainder);
    }
    */

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and 
    // alignment must be a multiple of 4 
    if(value_shift || bit_shift){
      int idx = num_bytes-4;
      uint32_t carry = 0;
      while(remainder){
        uint32_t v = reinterpret_cast<uint32_t*>(src)[idx/4];        
        //printf("VA : v(%x), (shift)%d, carry(%x), result(%x)\n", v, bit_shift, carry, ((v - value_shift) >> bit_shift) | carry);
        reinterpret_cast<uint32_t*>(dst)[idx/4] = ((v - value_shift) >> bit_shift) | carry;        
        carry = (v & ((1<<bit_shift)-1)) << (32 - bit_shift);
        remainder -= 4;
        idx-=4;
      }
    } else {
      while(remainder){
        int idx = num_bytes - remainder--;
        reinterpret_cast<uint8_t*>(dst)[idx] = reinterpret_cast<uint8_t*>(src)[idx];
      }
    }
  }
}

__global__ void copy_partitions(int num_columns, int num_partitions, int num_bufs,
                                uint8_t **src_bufs,
                                uint8_t **dst_bufs,
                                _column_info *buf_info)
{   
  int partition_index = blockIdx.x / num_columns;
  int column_index = blockIdx.x % num_columns;
  int t = threadIdx.x; 
  size_t buf_index = (partition_index * num_columns) + column_index; 
  int num_elements = buf_info[buf_index].num_elements;    
  int element_size = buf_info[buf_index].element_size;
  int stride = blockDim.x;
 
  int src_row_index = buf_info[buf_index].src_row_index;
  uint8_t *src = src_bufs[column_index];
  uint8_t *dst = dst_bufs[partition_index] + buf_info[buf_index].dst_offset;

  // copy, shifting offsets and validity bits as needed
  copy_buffer(dst, src, t, num_elements, element_size, src_row_index, stride, buf_info[buf_index].value_shift, buf_info[buf_index].bit_shift);
}

struct get_col_data {
  template<typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<T*>(col.begin<T>()));
    if(col.nullable()){      
      out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
  }

  template<typename T, std::enable_if_t<!cudf::is_fixed_width<T>() && std::is_same<T, cudf::string_view>::value>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    strings_column_view scv(col);
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<size_type*>(scv.offsets().begin<size_type>()));
    out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<int8_t*>(scv.chars().begin<int8_t>()));
    if(col.nullable()){      
      out_buf[col_index++] = reinterpret_cast<uint8_t*>(const_cast<bitmask_type*>(col.null_mask()));
    }
  }

  template<typename T, std::enable_if_t<!cudf::is_fixed_width<T>() && !std::is_same<T, cudf::string_view>::value>* = nullptr>
  void operator()(column_view const& col, size_type &col_index, uint8_t **out_buf)
  {
    CUDF_FAIL("unsupported type");    
  }
};


struct buf_size_functor {  
  _column_info const* ci;
  size_t operator() __device__ (int index)
  {
    //printf("Size : %d (%lu)\n", ci[index].buf_size, (uint64_t)(&ci[index]));
    return static_cast<size_t>(ci[index].buf_size);
  }
};

struct split_key_functor {
  int num_columns;
  int operator() __device__ (int t)
  {
    //printf("Key : %d (%d, %d)\n", t / num_columns, t , num_columns);
    return t / num_columns;
  }
};

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{ 
#if defined(__NEW_PATH)     
  if (input.num_columns() == 0) { return {}; }
  if(splits.size() > 0){
    CUDF_EXPECTS(splits.back() <= input.column(0).size(), "splits can't exceed size of input columns");
  }

  size_t num_root_columns = input.num_columns();

  // compute real # of columns (treating every child column individually), # of partitions
  // and total # of buffers
  size_t num_columns = 0;
  for(size_t idx=0; idx<num_root_columns; idx++){
    if(input.column(idx).type().id() == type_id::STRING){      
      num_columns+=2;
    } else {
      num_columns++;
    }
    if(input.column(idx).nullable()){
      num_columns++;
    }
  }
  size_t num_partitions = splits.size() + 1;
  size_t num_bufs = num_columns * num_partitions;  

  // compute total size of host-side temp data
  size_t indices_size = cudf::util::round_up_safe((splits.size() + 1) * 2 * sizeof(size_type), split_align);
  size_t src_column_info_size = cudf::util::round_up_safe(num_columns * sizeof(_src_column_info), split_align);  
  size_t buf_sizes_size = cudf::util::round_up_safe(num_partitions * sizeof(size_t), split_align);
  size_t column_infos_size = cudf::util::round_up_safe(num_bufs * sizeof(_column_info), split_align);
  size_t src_bufs_size = cudf::util::round_up_safe(num_columns * sizeof(uint8_t*), split_align);
  size_t dst_bufs_size = cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align);
  size_t total_temp_size = indices_size + src_column_info_size + buf_sizes_size + column_infos_size + src_bufs_size + dst_bufs_size;
  
  // allocate host
  std::vector<uint8_t> host_buf(total_temp_size);
  
  // distribute
  uint8_t *cur_h_buf = host_buf.data();
  size_type     *h_indices = reinterpret_cast<size_type*>(cur_h_buf);           cur_h_buf += indices_size;
  _src_column_info *h_column_src_info = reinterpret_cast<_src_column_info*>(cur_h_buf);  cur_h_buf += src_column_info_size;  
  size_t        *h_buf_sizes = reinterpret_cast<size_t*>(cur_h_buf);            cur_h_buf += buf_sizes_size;
  _column_info  *h_column_info = reinterpret_cast<_column_info*>(cur_h_buf);    cur_h_buf += column_infos_size;
  uint8_t       **h_src_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);          cur_h_buf += src_bufs_size;
  uint8_t       **h_dst_bufs = reinterpret_cast<uint8_t**>(cur_h_buf);


  // allocate device
  rmm::device_buffer device_buf{total_temp_size, stream, mr};

  // distribute
  uint8_t *cur_d_buf = reinterpret_cast<uint8_t*>(device_buf.data());
  size_type     *d_indices = reinterpret_cast<size_type*>(cur_d_buf);           cur_d_buf += indices_size;
  _src_column_info *d_column_src_info = reinterpret_cast<_src_column_info*>(cur_d_buf);  cur_d_buf += src_column_info_size;  
  size_t        *d_buf_sizes = reinterpret_cast<size_t*>(cur_d_buf);            cur_d_buf += buf_sizes_size;
  _column_info  *d_column_info = reinterpret_cast<_column_info*>(cur_d_buf);    cur_d_buf += column_infos_size;
  uint8_t       **d_src_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);          cur_d_buf += src_bufs_size;
  uint8_t       **d_dst_bufs = reinterpret_cast<uint8_t**>(cur_d_buf);

  // compute splits -> indices
  {
    size_type *indices = h_indices;
    *indices = 0;
    indices++;
    std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
      *indices = split;
      indices++;
      *indices = split;
      indices++; 
    });
    *indices = input.column(0).size();
  
    for (size_t i = 0; i < splits.size(); i++) {
      auto begin = h_indices[2 * i];
      auto end   = h_indices[2 * i + 1];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
    }   
  }
  
  // setup column types
  {
    int col_index = 0;
    for(size_t idx=0; idx<num_root_columns; idx++){
      if(input.column(idx).type().id() == type_id::STRING){
        strings_column_view scv(input.column(idx));

        h_column_src_info[col_index].type = type_id::INT32; 
        h_column_src_info[col_index].offsets = scv.offsets().begin<int>();
        h_column_src_info[col_index].is_offset_column = true;
        h_column_src_info[col_index].is_validity = false;
        col_index++;

        h_column_src_info[col_index].type = type_id::INT8;  // chars
        h_column_src_info[col_index].offsets = scv.offsets().begin<int>();
        h_column_src_info[col_index].is_offset_column = false;
        h_column_src_info[col_index].is_validity = false;        
        col_index++;
      } else {
        h_column_src_info[col_index].type = input.column(idx).type().id();
        h_column_src_info[col_index].offsets = nullptr;
        h_column_src_info[col_index].is_offset_column = false;
        h_column_src_info[col_index].is_validity = false; 
        col_index++;
      }

      // if we have validity
      if(input.column(idx).nullable()){
        h_column_src_info[col_index].type = type_id::INT32; 
        h_column_src_info[col_index].offsets = 0;
        h_column_src_info[col_index].is_offset_column = false;
        h_column_src_info[col_index].is_validity = true; 
        col_index++;
      }
    }
  }

  // HtoD indices and types to device
  cudaMemcpyAsync(d_indices, h_indices, indices_size + src_column_info_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // compute sizes of each column in each partition, including alignment.
  thrust::transform(rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_bufs),
    d_column_info,
    [num_columns, d_indices, d_column_src_info] __device__ (size_t t){      
      int split_index = t / num_columns;
      int column_index = t % num_columns;
      auto const& src_info = d_column_src_info[column_index];
      
      int row_index_start = d_indices[split_index * 2];
      int row_index_end = d_indices[split_index *2 + 1];
      int value_shift = 0;
      int bit_shift = 0;

      // if I am an offsets column, all my values need to be shifted
      if(src_info.is_offset_column){
        //printf("Value shift pre: %lu, %d\n", (uint64_t)src_info.offsets, row_index_start);
        value_shift = src_info.offsets[row_index_start];
        //printf("Value shift post: %d\n", value_shift);
      }
      // otherwise, if I have an associated offsets column adjust indices
      else if(src_info.offsets != nullptr){      
        row_index_start = src_info.offsets[row_index_start];
        row_index_end = src_info.offsets[row_index_end];        
      }      
      int num_elements = row_index_end - row_index_start;
      if(src_info.is_offset_column){
        num_elements++;
      }
      if(src_info.is_validity){
        bit_shift = row_index_start % 32;        
        num_elements = (num_elements + 31) / 32;
        row_index_start /= 32;
        row_index_end /= 32;
      }
      int element_size = cudf::type_dispatcher(data_type{src_info.type}, _size_of_helper{});
      size_t bytes = num_elements * element_size;
      //printf("Col %d, split %d (%d, %d), (%d, %lu), (%d, %d)\n", column_index, split_index, row_index_start, row_index_end, num_elements, bytes, value_shift, bit_shift);
      return _column_info{_round_up_safe(bytes, 64), num_elements, element_size, row_index_start, 0, value_shift, bit_shift};
    });

  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size + column_infos_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // compute total size of each partition
  {
    // key is split index
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), split_key_functor{static_cast<int>(num_columns)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0), buf_size_functor{d_column_info});

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream), 
      keys, keys + num_bufs, values, thrust::make_discard_iterator(), d_buf_sizes);
  }
  
  /*
  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  for(size_t idx=0; idx<num_partitions; idx++){
    printf("partition %lu : %lu\n", idx, h_buf_sizes[idx]);
  } 
  */ 

  // compute start offset for each output buffer
  {
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), split_key_functor{static_cast<int>(num_columns)});
    auto values = thrust::make_transform_iterator(thrust::make_counting_iterator(0), buf_size_functor{d_column_info});
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
      keys, keys + num_bufs, values, dst_offset_output_iterator{d_column_info}, 0);
  }

  // DtoH buf sizes and col info back to the host
  cudaMemcpyAsync(h_buf_sizes, d_buf_sizes, buf_sizes_size + column_infos_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  /*
  for(size_t idx=0; idx<h_buf_sizes.size(); idx++){
    printf("partition %lu : %lu\n", idx, h_buf_sizes[idx]);
  }
  */    
  // DtoH column info back to hose
  /*
  thrust::host_vector<_column_info> h_col_info(num_bufs);
  cudaMemcpyAsync(h_col_info.data(), d_col_info.data(), sizeof(_column_info) * num_bufs, cudaMemcpyDeviceToHost);
  */
  /*
  thrust::host_vector<size_t> h_dst_offsets(num_bufs);
  cudaMemcpyAsync(h_dst_offsets.data(), d_dst_offsets.data(), sizeof(size_t) * num_bufs, cudaMemcpyDeviceToHost);  
  */
  /*
  // debug  
  for(size_t idx=0; idx<h_col_info.size(); idx++){
    printf("size/offset : (%d, %d), %lu\n", h_col_info[idx].num_elements, h_col_info[idx].buf_size, h_dst_offsets[idx]);
  } 
  */   

  // allocate output partition buffers
  std::vector<rmm::device_buffer> out_buffers;
  out_buffers.reserve(num_partitions);
  std::transform(h_buf_sizes, h_buf_sizes + num_partitions, std::back_inserter(out_buffers), [stream, mr](size_t bytes){    
    return rmm::device_buffer{bytes, stream, mr};
  });

  // setup src buffers
  {
    size_type out_index = 0;  
    std::for_each(input.begin(), input.end(), [&out_index, &h_src_bufs](column_view const& col){
      cudf::type_dispatcher(col.type(), get_col_data{}, col, out_index, h_src_bufs);
    });  
  }

  // setup dst buffers
  {
    size_type out_index = 0;  
    std::for_each(out_buffers.begin(), out_buffers.end(), [&out_index, &h_dst_bufs](rmm::device_buffer& buf){
      h_dst_bufs[out_index++] = reinterpret_cast<uint8_t*>(buf.data());
    });
  }

  // HtoD src and dest buffers
  cudaMemcpyAsync(d_src_bufs, h_src_bufs, src_bufs_size + dst_bufs_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);  
    
  // copy.  1 block per buffer  
  {
    // scope_timer timer("kernel");
    constexpr int block_size = 512;
    copy_partitions<<<num_bufs, block_size, 0, stream>>>(num_columns, num_partitions, num_bufs,
                                                                d_src_bufs,
                                                                d_dst_bufs,
                                                                d_column_info);

    // TODO : put this after the column building step below to overlap work
    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // build the output.
  std::vector<contiguous_split_result> result;  
  result.reserve(num_partitions);
  size_t buf_index = 0;
  for(size_t idx=0; idx<num_partitions; idx++){
    std::vector<column_view> cols;
    cols.reserve(input.num_columns());    
    for(size_t s_idx=0; s_idx<num_root_columns; s_idx++){
      cudf::type_id id = input.column(s_idx).type().id();
      bool nullable = input.column(s_idx).nullable();
      
      if(id == type_id::STRING){
        cols.push_back(cudf::column_view{data_type{type_id::STRING}, 
                                          h_column_info[buf_index].num_elements-1, 
                                          nullptr,
                                          nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[idx] + h_column_info[buf_index+2].dst_offset) : nullptr,
                                          nullable ? UNKNOWN_NULL_COUNT : 0,
                                          0,
                                          {
                                            cudf::column_view{data_type{type_id::INT32}, h_column_info[buf_index].num_elements, 
                                            reinterpret_cast<void*>(h_dst_bufs[idx] + h_column_info[buf_index].dst_offset)},

                                            cudf::column_view{data_type{type_id::INT8}, h_column_info[buf_index+1].num_elements, 
                                            reinterpret_cast<void*>(h_dst_bufs[idx] + h_column_info[buf_index+1].dst_offset)}
                                          }});

        // cudf::test::print(cols.back());
        buf_index+=2;
      } else {
        cols.push_back(cudf::column_view{data_type{id}, 
                                         h_column_info[buf_index].num_elements, 
                                         reinterpret_cast<void*>(h_dst_bufs[idx] + h_column_info[buf_index].dst_offset),
                                         nullable ? reinterpret_cast<bitmask_type*>(h_dst_bufs[idx] + h_column_info[buf_index+1].dst_offset) : nullptr,
                                         nullable ? UNKNOWN_NULL_COUNT : 0
                                         });
        buf_index++;
      }      
      if(nullable){
        buf_index++;
      }
    }
    result.push_back(contiguous_split_result{cudf::table_view{cols}, std::make_unique<rmm::device_buffer>(std::move(out_buffers[idx]))});
  }    

  return std::move(result);
#else
  auto subtables = cudf::split(input, splits);
  
  // optimization : for large numbers of splits this allocation can dominate total time
  //                spent if done inside alloc_and_copy().  so we'll allocate it once
  //                and reuse it.
  //
  //                benchmark:        1 GB data, 10 columns, 256 splits.
  //                no optimization:  106 ms (8 GB/s)
  //                optimization:     20 ms (48 GB/s)
  rmm::device_vector<column_split_info> device_split_info(input.num_columns());  

  std::vector<contiguous_split_result> result;
  std::transform(subtables.begin(),
                 subtables.end(),
                 std::back_inserter(result),
                 [mr, stream, &device_split_info](table_view const& t) {
                   return alloc_and_copy(t, device_split_info, mr, stream);
                 });                 

  return result;
#endif  
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
