
#include <cudf/bitmask/bitmask.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace detail {

// These will be written together to global memory as the output of a null mask operation.
// Assumes output null_mask buffer is a temporary buffer that can be overwritten.
struct null_mask_chunk {
  cudf::bitmask_type word;
  cudf::size_type index;
  cudf::size_type valid_count;

  // num_chunks and src_size must be >= 0
  template <typename BitMaskWordFunc>
  static __device__ null_mask_chunk load(cudf::size_type current_chunk,
                                         BitMaskWordFunc&& src_null_mask,
                                         cudf::size_type src_offset,
                                         cudf::size_type src_size)
  {
    auto chunk_bit_begin = current_chunk * static_cast<cudf::size_type>(sizeof(bitmask_type) * 8);
    constexpr auto num_chunk_bits = static_cast<cudf::size_type>(sizeof(bitmask_type) * 8);

    auto src_bit_begin       = src_offset + chunk_bit_begin;
    auto const num_src_words = (src_size + (num_chunk_bits - 1)) / num_chunk_bits;

    auto leading_word_index  = src_bit_begin / num_chunk_bits;
    auto trailing_word_index = (src_bit_begin + (num_chunk_bits - 1)) / num_chunk_bits;

    if (trailing_word_index < num_src_words) [[likely]] {
      auto leading_bits  = src_null_mask(leading_word_index);
      auto trailing_bits = src_null_mask(trailing_word_index);
      auto bit_shift     = src_bit_begin % num_chunk_bits;
      auto merged = (cudf::bitmask_type)__funnelshift_r(leading_bits, trailing_bits, bit_shift);
      auto valid_count = (cudf::size_type)__popc(merged);
      return null_mask_chunk{merged, current_chunk, valid_count};
    } else {
      auto leading_bits     = src_null_mask(leading_word_index);
      auto bit_shift        = src_bit_begin % num_chunk_bits;
      auto num_discard_bits = (src_bit_begin + num_chunk_bits) - (src_offset + src_size);
      auto mask             = (~bitmask_type{0}) >> num_discard_bits;
      auto output           = (leading_bits >> bit_shift) & mask;
      auto valid_count      = (cudf::size_type)__popc(output);
      return null_mask_chunk{output, current_chunk, valid_count};
    }
  }
};

template <int block_size, typename Binop>
CUDF_KERNEL void chunked_bitmask_binop(Binop op,
                                       device_span<bitmask_type> destination,
                                       device_span<bitmask_type const* const> source,
                                       device_span<size_type const> source_begin_bits,
                                       size_type source_size_bits,
                                       size_type num_chunks,
                                       size_type* valid_count_ptr)
{
  auto const tid              = cudf::detail::grid_1d::global_thread_id();
  cudf::size_type valid_count = 0;

  // for(auto i = tid;)
  // [ ] can we reduce the number of blocks so we can have more warps per block?

  // [ ] let the CPU sum the valid counts from each block

  //[ ] use duff's device to unroll the sources
  //   switch(src.size() / 16){
  // [ ] use function to load
  // 16 pairs + reduce
  // 8 pairs + reduce
  // 4 pairs + reduce
  // 2 pairs + reduce
  // 1 pair + reduce
  //   }

  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_type block_valid_count = BlockReduce(temp_storage).Sum(valid_count);

  if (threadIdx.x == 0) { atomicAdd(valid_count_ptr, block_valid_count); }
}

// to embed in transform kernel

// write output to uint32_t
// once full or at end, write to global memory; using null_mask_chunk

}  // namespace detail
}  // namespace cudf
