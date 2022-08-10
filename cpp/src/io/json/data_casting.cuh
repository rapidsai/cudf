/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

 #include <io/utilities/parsing_utils.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>


namespace cudf::io::json::experimental {

/**
 * @brief Decodes a numeric value base on templated cudf type T with specified
 * base.
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed numeric value
 */
 template <typename T, int base>
 __inline__ __device__ T decode_value(const char* begin,
                                      uint64_t end,
                                      parse_options_view const& opts)
 {
   return cudf::io::parse_numeric<T, base>(begin, end, opts);
 }
 
 /**
  * @brief Decodes a numeric value base on templated cudf type T
  *
  * @param[in] begin Beginning of the character string
  * @param[in] end End of the character string
  * @param opts The global parsing behavior options
  *
  * @return The parsed numeric value
  */
 template <typename T,
           std::enable_if_t<!cudf::is_timestamp<T>() and !cudf::is_duration<T>()>* = nullptr>
 __inline__ __device__ T decode_value(const char* begin,
                                      const char* end,
                                      parse_options_view const& opts)
 {
   return cudf::io::parse_numeric<T>(begin, end, opts);
 }
 
 template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
 __inline__ __device__ T decode_value(char const* begin,
                                      char const* end,
                                      parse_options_view const& opts)
 {
   return to_timestamp<T>(begin, end, opts.dayfirst);
 }
 
 template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
 __inline__ __device__ T decode_value(char const* begin, char const* end, parse_options_view const&)
 {
   return to_duration<T>(begin, end);
 }
 
 // The purpose of these is merely to allow compilation ONLY
 template <>
 __inline__ __device__ cudf::string_view decode_value(const char*,
                                                      const char*,
                                                      parse_options_view const&)
 {
   return cudf::string_view{};
 }
 
 template <>
 __inline__ __device__ cudf::dictionary32 decode_value(const char*,
                                                       const char*,
                                                       parse_options_view const&)
 {
   return cudf::dictionary32{};
 }
 
 template <>
 __inline__ __device__ cudf::list_view decode_value(const char*,
                                                    const char*,
                                                    parse_options_view const&)
 {
   return cudf::list_view{};
 }
 template <>
 __inline__ __device__ cudf::struct_view decode_value(const char*,
                                                      const char*,
                                                      parse_options_view const&)
 {
   return cudf::struct_view{};
 }
 
 template <>
 __inline__ __device__ numeric::decimal32 decode_value(const char*,
                                                       const char*,
                                                       parse_options_view const&)
 {
   return numeric::decimal32{};
 }
 
 template <>
 __inline__ __device__ numeric::decimal64 decode_value(const char*,
                                                       const char*,
                                                       parse_options_view const&)
 {
   return numeric::decimal64{};
 }
 
 template <>
 __inline__ __device__ numeric::decimal128 decode_value(const char*,
                                                        const char*,
                                                        parse_options_view const&)
 {
   return numeric::decimal128{};
 }
 
  struct ConvertFunctor {
    /**
     * @brief Template specialization for operator() for types whose values can be
     * convertible to a 0 or 1 to represent false/true. The converting is done by
     * checking against the default and user-specified true/false values list.
     *
     * It is handled here rather than within convertStrToValue() as that function
     * is used by other types (ex. timestamp) that aren't 'booleable'.
     */
    template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
    __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                        char const* end,
                                                        void* output_column,
                                                        cudf::size_type row,
                                                        const parse_options_view& opts)
    {
      T& value{static_cast<T*>(output_column)[row]};
  
      value = [&opts, end, begin]() -> T {
        // Check for user-specified true/false values
        auto const len = static_cast<size_t>(end - begin);
        if (serialized_trie_contains(opts.trie_true, {begin, len})) { return 1; }
        if (serialized_trie_contains(opts.trie_false, {begin, len})) { return 0; }
        return decode_value<T>(begin, end, opts);
      }();
  
      return true;
    }
  
    /**
     * @brief Dispatch for floating points, which are set to NaN if the input
     * is not valid. In such case, the validity mask is set to zero too.
     */
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                        char const* end,
                                                        void* out_buffer,
                                                        size_t row,
                                                        parse_options_view const& opts)
    {
      T const value                    = decode_value<T>(begin, end, opts);
      static_cast<T*>(out_buffer)[row] = value;
  
      return !std::isnan(value);
    }
  
    /**
     * @brief Default template operator() dispatch specialization all data types
     * (including wrapper types) that is not covered by above.
     */
    template <typename T,
              std::enable_if_t<!std::is_floating_point_v<T> and !std::is_integral_v<T>>* = nullptr>
    __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                        char const* end,
                                                        void* output_column,
                                                        cudf::size_type row,
                                                        const parse_options_view& opts)
    {
      static_cast<T*>(output_column)[row] = decode_value<T>(begin, end, opts);
  
      return true;
    }
  };

template <typename str_tuple_it>
rmm::device_uvector<thrust::pair<const char*, size_type>> coalesce_input(
  str_tuple_it str_tuples, size_type col_size, rmm::cuda_stream_view stream)
{
  auto result = rmm::device_uvector<thrust::pair<const char*, size_type>>(col_size, stream);
  thrust::copy_n(rmm::exec_policy(stream), str_tuples, col_size, result.begin());
  return result;
}

template <typename str_tuple_it, typename B>
std::unique_ptr<column> parse_data(str_tuple_it str_tuples,
                                   size_type col_size,
                                   data_type col_type,
                                   B&& null_mask,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto parse_opts = parse_options{',', '\n', '\"', '.'};

  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    auto const strings_span = coalesce_input(str_tuples, col_size, stream);
    return make_strings_column(strings_span, stream);
  }

  auto out_col = make_fixed_width_column(
    col_type, col_size, std::move(null_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  auto output_dv_ptr = mutable_column_device_view::create(*out_col, stream);

  // use existing code (`ConvertFunctor`) to convert values
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     col_size,
                     [str_tuples, col = *output_dv_ptr, opts = parse_opts.view()] __device__(size_type row_idx) {
                      auto const in = str_tuples[row_idx];
                       cudf::type_dispatcher(column_types[desc.column],
                        ConvertFunctor{},
                        in.first,
                        in.first + in.second,
                        col.data(),
                        row_idx,
                        opts)
                     });

  return out_col;
}

}  // namespace cudf::io::json::experimental
