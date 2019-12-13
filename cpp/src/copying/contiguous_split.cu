#include <cudf/cudf.h>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/bit.hpp>

#include <numeric>

namespace cudf {

namespace experimental {

namespace detail {

namespace {

template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_in_place_kernel( column_device_view const in,
                           mutable_column_device_view out)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;      

   // begin/end indices for the column data
   size_type begin = 0;
   size_type end = in.size();
   // warp indices.  since 1 warp == 32 threads == sizeof(bit_mask_t) * 8,
   // each warp will process one (32 bit) of the validity mask via
   // __ballot_sync()
   size_type warp_begin = cudf::word_index(begin);
   size_type warp_end = cudf::word_index(end-1);      

   // lane id within the current warp   
   const int lane_id = threadIdx.x % cudf::experimental::detail::warp_size;
   
   // current warp.
   size_type warp_cur = warp_begin + warp_id;   
   size_type index = tid;
   while(warp_cur <= warp_end){
      bool in_range = (index >= begin && index < end);
            
      bool valid = true;
      if(has_validity){
         valid = in_range && in.is_valid(index);
      }
      if(in_range){
         out.element<T>(index) = in.element<T>(index);
      }
      
      // update validity      
      if(has_validity){
         // the final validity mask for this warp 
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && in_range);
         // only one guy in the warp needs to update the mask and count
         if(lane_id == 0){            
            out.set_mask_word(warp_cur, warp_mask);            
         }
      }            

      // next grid
      warp_cur += warps_per_grid;
      index += block_size * gridDim.x;
   }      
}

static constexpr size_t split_align = 8;

struct column_buf_size_functor {   
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   std::pair<size_t, size_t> operator()(std::pair<size_t, size_t> sizes, column_view const& c)
   {
      CUDF_FAIL("contiguous_split for strings not implemented yet");
      return std::pair<size_t, size_t>(0, 0);
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   std::pair<size_t, size_t> operator()(std::pair<size_t, size_t> sizes, column_view const& c)
   {
      size_t data_size = sizes.first + (cudf::util::div_rounding_up_safe(c.size() * sizeof(T), split_align) * split_align);
      size_t validity_size = sizes.second + (c.nullable() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
      return std::pair<size_t, size_t>(data_size, validity_size);
   }
};

cudaStream_t stream1, stream2;

__global__ void _copy_offset_bitmask(bitmask_type *__restrict__ destination,
                                    bitmask_type const *__restrict__ source,
                                    size_type source_begin_bit,
                                    size_type source_end_bit,
                                    size_type number_of_mask_words) {
  for (size_type destination_word_index = threadIdx.x + blockIdx.x * blockDim.x;
       destination_word_index < number_of_mask_words;
       destination_word_index += blockDim.x * gridDim.x) {
    size_type source_word_index =
        destination_word_index + word_index(source_begin_bit);
    bitmask_type curr_word = source[source_word_index];
    bitmask_type next_word = 0;
    if ((word_index(source_begin_bit) != 0) &&
        (word_index(source_end_bit) >
          word_index(source_begin_bit +
            destination_word_index * cudf::detail::size_in_bits<bitmask_type>()))) {
      next_word = source[source_word_index + 1];
    }
    bitmask_type write_word =
      __funnelshift_r(curr_word, next_word, source_begin_bit);
    destination[destination_word_index] = write_word;
  }
}

// Create a bitmask from a specific range
void _copy_bitmask(bitmask_type *dest_mask, bitmask_type const *mask, size_type begin_bit,
                                size_type end_bit, cudaStream_t stream) {
  CUDF_EXPECTS(begin_bit >= 0, "Invalid range.");
  CUDF_EXPECTS(begin_bit <= end_bit, "Invalid bit range.");  
  auto num_bytes = bitmask_allocation_size_bytes(end_bit - begin_bit);  
    
   auto number_of_mask_words = cudf::util::div_rounding_up_safe(
      static_cast<size_t>(end_bit - begin_bit),
      cudf::detail::size_in_bits<bitmask_type>());
   
   if (begin_bit == 0) {
      cudaMemcpyAsync(dest_mask, mask, num_bytes, cudaMemcpyDeviceToDevice, stream); 
  } else {     
      cudf::experimental::detail::grid_1d config(number_of_mask_words, 256);
      _copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0,
                           stream>>>(dest_mask, mask, begin_bit, end_bit, number_of_mask_words);
  }
   CUDA_CHECK_LAST();    
}

struct column_copy_functor {   
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, char*& dst, std::vector<column_view>& out_cols) 
   {
      CUDF_FAIL("contiguous_split for strings not implemented yet");
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, char*& dst, std::vector<column_view>& out_cols)
   {
      // there's some unnecessary recomputation of sizes happening here, but it really shouldn't affect much.      
      auto sizes = column_buf_size_functor{}.operator()<T>(std::pair<size_t, size_t>(0, 0), in);
      size_t data_size = sizes.first;
      size_t validity_size = sizes.second;
   
      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + data_size);

      // increment working buffer
      dst += (data_size + validity_size);      

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};
                                          
      cudaMemcpyAsync(data, in.data<T>(), data_size, cudaMemcpyDeviceToDevice, stream1);
      if(in.nullable()){           
         _copy_bitmask(validity, in.null_mask(), in.offset(), in.offset() + in.size(), (cudaStream_t)stream2);
      }
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, UNKNOWN_NULL_COUNT };             
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // extremely slow because a.) the typical use case here will involve huge numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies.
      //
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.            
      //      
      /*
      column_view   in_wrapped{in.type(), in.size(), in.head<T>(), 
                               in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                               in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){               
         copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           *mutable_column_device_view::create(mcv));         
      } else {
         copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           *mutable_column_device_view::create(mcv));
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);                  
      */

      out_cols.push_back(mcv);
   }
};

#include <inttypes.h>
contiguous_split_result alloc_and_copy(cudf::table_view const& t, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{      
   // compute sizes  
   auto sizes = std::accumulate(t.begin(), t.end(), std::pair<size_t, size_t>(0, 0), [](std::pair<size_t, size_t> sizes, cudf::column_view const& c){
      return cudf::experimental::type_dispatcher(c.type(), column_buf_size_functor{}, sizes, c);
   });
   size_t data_size = sizes.first;
   size_t validity_size = sizes.second;   

   // allocate 
   auto device_buf = std::make_unique<rmm::device_buffer>(rmm::device_buffer{data_size + validity_size, stream1, mr});   
   char *buf = static_cast<char*>(device_buf->data());

   // copy
   std::vector<column_view> out_cols;
   out_cols.reserve(t.num_columns());
   std::for_each(t.begin(), t.end(), [&out_cols, &buf](cudf::column_view const& c){
      cudf::experimental::type_dispatcher(c.type(), column_copy_functor{}, c, buf, out_cols);
   });
   
   return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};
}

}; // anonymous namespace

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{    
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   printf("%" PRIx64 ", %" PRIx64 "\n", (long unsigned int)stream1, (long unsigned int)stream2);

   auto subtables = cudf::experimental::split(input, splits);      

   std::vector<contiguous_split_result> result;
   std::transform(subtables.begin(), subtables.end(), std::back_inserter(result), [mr, stream](table_view const& t) { 
      return alloc_and_copy(t, mr, stream);
   });

   cudaStreamSynchronize(stream1);
   cudaStreamSynchronize(stream2);
   
   return result;
}

}; // namespace detail

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr)
{    
   return cudf::experimental::detail::contiguous_split(input, splits, mr, (cudaStream_t)0);   
}

}; // namespace experimental

}; // namespace cudf
