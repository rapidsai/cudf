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

template <typename S>
__device__ inline S round_up_safe_nothrow(S number_to_round, S modulus) {
    auto remainder = number_to_round % modulus;
    if (remainder == 0) { return number_to_round; }
    auto rounded_up = number_to_round - remainder + modulus;    
    return rounded_up;
}

// Computes required allocation size of a bitmask
__device__ std::size_t bitmask_allocation_size_bytes_nothrow(size_type number_of_bits,
                                          std::size_t padding_boundary) {  
  auto necessary_bytes =
      cudf::util::div_rounding_up_safe<size_type>(number_of_bits, CHAR_BIT);

  auto padded_bytes =
      padding_boundary * cudf::util::div_rounding_up_safe<size_type>(
                             necessary_bytes, padding_boundary);
  return padded_bytes;
}


/**
 * @brief Copies contents of `in` to `out`.  Copies validity if present
 * but does not compute null count.
 *  
 * @param in column_view to copy from
 * @param out mutable_column_view to copy to.
 */
template <size_type block_size, typename T, bool has_validity>
__launch_bounds__(block_size)
__global__
void copy_in_place_kernel( column_device_view const in,
                           size_type validity_size,
                           mutable_column_device_view out,
                           T val_subtract)
{
   const size_type tid = threadIdx.x + blockIdx.x * block_size;
   const int warp_id = tid / cudf::experimental::detail::warp_size;
   const size_type warps_per_grid = gridDim.x * block_size / cudf::experimental::detail::warp_size;      

   // begin/end indices for the column data
   size_type begin = 0;      
   size_type end = in.size();   
   size_type validity_end = validity_size;   
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
      bool validity_in_range = (index >= begin && index < validity_end);
      bool valid = true;      
      if(has_validity){         
         valid = validity_in_range && in.is_valid(index);
      }

      bool in_range = (index >= begin && index < end);
      if(in_range){         
         out.element<T>(index) = in.element<T>(index) - val_subtract;
      }
      
      // update validity      
      if(has_validity && validity_in_range){
         // the final validity mask for this warp 
         int warp_mask = __ballot_sync(0xFFFF'FFFF, valid && validity_in_range);
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

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr size_t split_align = 64;

struct column_split_info {
   size_t   data_size;     // size of the data
   size_t   validity_size; // validity vector size
   
   size_t   offsets_size;  // (strings only) size of offset column
   size_t   chars_size;    // (strings only) # of chars in the column
   size_t   chars_offset;  // (strings only) offset from head of chars data
};

/**
 * @brief Functor called by the `type_dispatcher` to incrementally compute total
 * memory buffer size needed to allocate a contiguous copy of all columns within
 * a source table. 
 */
struct column_buffer_size_functor {
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {
      // this has already been precomputed in an earlier step      
      return split_info.data_size + split_info.validity_size + split_info.offsets_size;
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   size_t operator()(column_view const& c, column_split_info &split_info)
   {      
      split_info.data_size = cudf::util::round_up_safe(c.size() * sizeof(T), split_align);  
      split_info.validity_size = (c.nullable() ? cudf::bitmask_allocation_size_bytes(c.size(), split_align) : 0);
      return split_info.data_size + split_info.validity_size;
   }
};

/**
 * @brief Functor called by the `type_dispatcher` to copy a column into a contiguous
 * buffer of output memory. 
 * 
 * Used for copying each column in a source table into one contiguous buffer of memory.
 */
struct column_copy_functor {
   template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {            
      strings_column_view strings_c(in);      

      // outgoing pointers
      char* chars_buf = dst;
      bitmask_type* validity_buf = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);
      char* offsets_buf = dst + split_info.data_size + split_info.validity_size;

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size + split_info.offsets_size);

      // 2 kernel calls. 1 to copy offsets and validity, and another to copy chars            
      
      // copy offsets and validity
      column_view offsets_col = strings_c.offsets();
      mutable_column_view temp_offsets_and_validity{
                              offsets_col.type(), offsets_col.size(), offsets_buf,
                              validity_buf, validity_buf == nullptr ? UNKNOWN_NULL_COUNT : 0,
                              0 };
      {         
         // contruct a column which wraps the validity vector and the offsets from the child column. 
         // this is weird but it removes an extra kernel call. however, since the length of the offsets column
         // is always 1 greater than the # of strings, the validity vector will be short by 1. the kernel will have to
         // compensate for that. 
         CUDF_EXPECTS(in.size() == offsets_col.size()-1, "Expected offsets column to be the same size as parent");
         CUDF_EXPECTS(in.offset() == offsets_col.offset(), "Expected offsets column offset to be the same as parent");
         CUDF_EXPECTS(offsets_col.type() == cudf::data_type(INT32), "Expected offsets column type to be int32");
         column_view in_offsets_and_validity{
                                 offsets_col.type(), offsets_col.size(), offsets_col.head<int32_t>(),
                                 in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                                 in.offset()};
         
         cudf::size_type num_els = cudf::util::round_up_safe(strings_c.offsets().size(), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};         
         if(in.nullable()){
            copy_in_place_kernel<block_size, size_type, true><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity), 
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         } else {
            copy_in_place_kernel<block_size, size_type, false><<<grid.num_blocks, block_size, 0, 0>>>(
                              *column_device_view::create(in_offsets_and_validity),
                              in.size(),  // validity vector length
                              *mutable_column_device_view::create(temp_offsets_and_validity), split_info.chars_offset);
         }
      }

      // get the chars column directly instead of calling .chars(), since .chars() will end up
      // doing gpu work we specifically want to avoid.
      column_view chars_col = in.child(strings_column_view::chars_column_index);

      // copy chars
      mutable_column_view out_chars{chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_buf};      
      {         
         CUDF_EXPECTS(!chars_col.nullable(), "Expected input chars column to not be nullable");
         CUDF_EXPECTS(chars_col.offset() == 0, "Expected input chars column to have an offset of 0");
         column_view in_chars{ chars_col.type(), static_cast<size_type>(split_info.chars_size), chars_col.data<char>() + split_info.chars_offset };
                                 
         cudf::size_type num_els = cudf::util::round_up_safe(static_cast<size_type>(split_info.chars_size), cudf::experimental::detail::warp_size);
         constexpr int block_size = 256;
         cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};         
         copy_in_place_kernel<block_size, char, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_chars),
                           split_info.chars_size,
                           *mutable_column_device_view::create(out_chars), 0);
      }

      // construct output string column_view.  offsets and validity have been glued together so
      // we have to rearrange things a bit.      
      column_view out_offsets{strings_c.offsets().type(), strings_c.offsets().size(), offsets_buf};
      
      out_cols.push_back(column_view(in.type(), in.size(), nullptr,
                                     validity_buf, UNKNOWN_NULL_COUNT, 0,
                                     { out_offsets, out_chars }));
   }

   template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
   void operator()(column_view const& in, column_split_info const& split_info, char*& dst, std::vector<column_view>& out_cols)
   {     
      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = split_info.validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + split_info.data_size);

      // increment working buffer
      dst += (split_info.data_size + split_info.validity_size);

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // extremely slow because a.) the typical use case here will involve huge numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies.
      //
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.            
      //            
      // Remove this hack once rapidsai/cudf#3600 is fixed.
      column_view   in_wrapped{in.type(), in.size(), in.head<T>(), 
                               in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                               in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){               
         copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           in.size(),
                           *mutable_column_device_view::create(mcv), 0);         
      } else {
         copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_wrapped), 
                           in.size(),
                           *mutable_column_device_view::create(mcv), 0);
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);                 

      out_cols.push_back(mcv);
   }
};

/**
 * @brief Creates a contiguous_split_result object which contains a deep-copy of the input
 * table_view into a single contiguous block of memory. 
 * 
 * The table_view contained within the contiguous_split_result will pass an expect_tables_equal()
 * call with the input table.  The memory referenced by the table_view and its internal column_views
 * is entirely contained in single block of memory.
 */
contiguous_split_result alloc_and_copy(cudf::table_view const& t, thrust::device_vector<column_split_info>& device_split_info, rmm::mr::device_memory_resource* mr, cudaStream_t stream)      
{            
   // preprocess column sizes for string columns.  the idea here is this:
   // - determining string lengths involves reaching into device memory to look at offsets, which is slow.
   // - contiguous_split() is typically used in situations with very large numbers of output columns, exaggerating
   //   the problem.
   // - so rather than reaching into device memory once per column (in column_buffer_size_functor), 
   //   we are doing it once per split (for all string columns in the split).  For an example case of a table with 
   //   512 columns split 256 ways, that reduces our number of trips to/from the gpu from 128k -> 256
   
   // build a list of all the offset columns and their indices for all input string columns and put them on the gpu
   //
   // i'm using this pair structure instead of thrust::tuple because using tuple somehow causes the cudf::column_device_view
   // default constructor to get called (compiler error) when doing the assignment to device_offset_columns below
   thrust::host_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> offset_columns;
   offset_columns.reserve(t.num_columns());  // worst case
   size_type column_index = 0;
   std::for_each(t.begin(), t.end(), [&offset_columns, &column_index](cudf::column_view const& c){
      if(c.type().id() == STRING){
         // constructing device view from the offsets column only, because doing so for the entire
         // strings_column_view will result in memory allocation/cudaMemcpy() calls, which would
         // defeat the whole purpose of this step.
         cudf::column_device_view cdv((strings_column_view(c)).offsets(), 0, 0);
         offset_columns.push_back(thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>(
                  thrust::pair<size_type, bool>(column_index, c.nullable()), cdv));
      }
      column_index++;
   });   
   thrust::device_vector<thrust::pair<thrust::pair<size_type, bool>, cudf::column_device_view>> device_offset_columns = offset_columns;   
     
   // compute column sizes for all string columns   
   auto *sizes_p = device_split_info.data().get();   
   thrust::for_each(rmm::exec_policy(stream)->on(stream), device_offset_columns.begin(), device_offset_columns.end(),
      [sizes_p] __device__ (auto column_info){
         size_type                  col_index = column_info.first.first;
         bool                       include_validity = column_info.first.second;
         cudf::column_device_view   col = column_info.second;
         size_type                  num_elements = col.size()-1;

         size_t align = split_align;

         auto num_chars = col.data<int32_t>()[num_elements] - col.data<int32_t>()[0];         
         sizes_p[col_index].data_size = round_up_safe_nothrow(static_cast<size_t>(num_chars), align);         
         // can't use cudf::bitmask_allocation_size_bytes() because it throws
         sizes_p[col_index].validity_size = include_validity ? bitmask_allocation_size_bytes_nothrow(num_elements, align) : 0;                  
         // can't use cudf::util::round_up_safe() because it throws
         sizes_p[col_index].offsets_size = round_up_safe_nothrow(col.size() * sizeof(size_type), align);
         sizes_p[col_index].chars_size = num_chars;
         sizes_p[col_index].chars_offset = col.data<int32_t>()[0];
      }
   );
   
   // copy sizes back from gpu. entries from non-string columns are uninitialized at this point.
   thrust::host_vector<column_split_info> split_info = device_split_info;  
     
   // compute the rest of the column sizes (non-string columns, and total buffer size)
   size_t total_size = 0;
   column_index = 0;
   std::for_each(t.begin(), t.end(), [&total_size, &column_index, &split_info](cudf::column_view const& c){   
      total_size += cudf::experimental::type_dispatcher(c.type(), column_buffer_size_functor{}, c, split_info[column_index]);
      column_index++;
   });

   // allocate
   auto device_buf = std::make_unique<rmm::device_buffer>(total_size, stream, mr);
   char *buf = static_cast<char*>(device_buf->data());

   // copy (this would be cleaner with a std::transform, but there's an nvcc compiler issue in the way)   
   std::vector<column_view> out_cols;
   out_cols.reserve(t.num_columns());
   column_index = 0;   
   std::for_each(t.begin(), t.end(), [&out_cols, &buf, &column_index, &split_info](cudf::column_view const& c){
      cudf::experimental::type_dispatcher(c.type(), column_copy_functor{}, c, split_info[column_index], buf, out_cols);
      column_index++;
   });   
   
   return contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)};   
}

}; // anonymous namespace

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
{   
   auto subtables = cudf::experimental::split(input, splits);

   // optimization : for large #'s of splits this allocation can dominate total time
   //                spent if done inside alloc_and_copy().  so we'll allocate it once
   //                and reuse it.
   // 
   //                benchmark:        1 GB data, 10 columns, 256 splits.
   //                no optimization:  106 ms (8 GB/s)
   //                optimization:     20 ms (48 GB/s)
   thrust::device_vector<column_split_info> device_split_info(input.num_columns());

   std::vector<contiguous_split_result> result;
   std::transform(subtables.begin(), subtables.end(), std::back_inserter(result), [mr, stream, &device_split_info](table_view const& t) { 
      return alloc_and_copy(t, device_split_info, mr, stream);
   });

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
