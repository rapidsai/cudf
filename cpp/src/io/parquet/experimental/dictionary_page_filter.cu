/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/parquet_gpu.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <optional>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_error;
using parquet::detail::PageInfo;

namespace {

// Decode kernel parameters
auto constexpr int96_size        = 12;   ///< Size of INT96 physical type
auto constexpr decode_block_size = 128;  ///< Must be a multiple of warp_size
/// Maximum number of literals to evaluate while decoding column dictionaries
auto constexpr max_inline_literals = 2;

// cuCollections static set parameters
using key_type = cudf::size_type;                  ///< Using data indices (size_type) as set keys
auto constexpr empty_key_sentinel = key_type{-1};  ///< We will never encounter a -1 index.
auto constexpr set_cg_size        = 1;             ///< Cooperative group size for cuco::static_set
auto constexpr bucket_size        = 1;             ///< Number of buckets per set slot
auto constexpr occupancy_factor = 70;  ///< cuCollections suggests targeting a 70% occupancy factor

using storage_type     = cuco::bucket_storage<key_type,
                                          bucket_size,
                                          cuco::extent<std::size_t>,
                                          cudf::detail::cuco_allocator<char>>;
using storage_ref_type = typename storage_type::ref_type;
using bucket_type      = typename storage_type::bucket_type;

template <typename T>
struct insert_hash_functor {
  cudf::device_span<T> const decoded_data;
  uint32_t const seed;
  __device__ constexpr auto operator()(key_type idx) const noexcept
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(decoded_data[idx]);
  }
};

template <typename T>
struct insert_equality_functor {
  cudf::device_span<T> const decoded_data;
  __device__ constexpr bool operator()(key_type lhs_idx, key_type rhs_idx) const noexcept
  {
    return decoded_data[lhs_idx] == decoded_data[rhs_idx];
  }
};

template <typename T>
struct query_hash_functor {
  uint32_t const seed;
  __device__ constexpr auto operator()(T const& key) const noexcept
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(key);
  }
};

template <typename T>
struct query_equality_functor {
  cudf::device_span<T> const decoded_data;
  __device__ constexpr bool operator()(T const& lhs, key_type rhs) const noexcept
  {
    return lhs == decoded_data[rhs];
  }
};

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
query_dictionaries(cudf::device_span<T> decoded_data,
                   cudf::device_span<bool*> results,
                   ast::generic_scalar_device_view const* scalars,
                   ast::ast_operator const* operators,
                   bucket_type* const set_storage,
                   cudf::size_type const* set_offsets,
                   cudf::size_type const* value_offsets,
                   cudf::size_type total_row_groups,
                   parquet::Type physical_type)
{
  namespace cg          = cooperative_groups;
  auto const scalar_idx = cg::this_grid().block_rank();
  auto const scalar     = scalars[scalar_idx];
  auto result           = results[scalar_idx];

  using equality_fn_type    = query_equality_functor<T>;
  using hash_fn_type        = query_hash_functor<T>;
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  auto const group = cg::this_thread_block();
  for (auto set_idx = group.thread_rank(); set_idx < total_row_groups; set_idx += group.size()) {
    // If the set is empty (no dictionary page data), then skip the dictionary page filter
    if (set_offsets[set_idx + 1] - set_offsets[set_idx] == 0) {
      result[set_idx] = operators[scalar_idx] == ast::ast_operator::EQUAL;
      continue;
    }

    storage_ref_type const storage_ref{set_offsets[set_idx + 1] - set_offsets[set_idx],
                                       set_storage + set_offsets[set_idx]};

    auto hash_set_ref = cuco::static_set_ref{cuco::empty_key{empty_key_sentinel},
                                             equality_fn_type{decoded_data},
                                             probing_scheme_type{hash_fn_type{}},
                                             cuco::thread_scope_block,
                                             storage_ref};

    auto set_find_ref         = hash_set_ref.rebind_operators(cuco::contains);
    auto literal_value        = scalar.value<T>();
    auto const num_set_values = value_offsets[set_idx + 1] - value_offsets[set_idx];
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      if (physical_type == parquet::Type::INT96) {
        auto const int128_key = static_cast<__int128_t>(scalar.value<int64_t>());
        auto probe_key = cudf::string_view{reinterpret_cast<char const*>(&int128_key), int96_size};
        literal_value  = probe_key;
      }
    }

    if (operators[scalar_idx] == ast::ast_operator::NOT_EQUAL) {
      result[set_idx] = num_set_values == 1 and set_find_ref.contains(literal_value);
    } else {
      result[set_idx] = set_find_ref.contains(literal_value);
    }
  }
}

__global__ void build_string_dictionaries(PageInfo const* pages,
                                          cudf::device_span<cudf::string_view> decoded_data,
                                          bucket_type* const set_storage,
                                          cudf::size_type const* set_offsets,
                                          cudf::size_type const* value_offsets,
                                          cudf::size_type total_row_groups,
                                          cudf::size_type num_dictionary_columns,
                                          cudf::size_type dictionary_col_idx,
                                          kernel_error::pointer error)
{
  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const row_group_idx =
    (cg::this_grid().block_rank() * warp.meta_group_size()) + warp.meta_group_rank();

  if (row_group_idx > total_row_groups) { return; }

  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page_data_size == 0) { return; }

  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage + set_offsets[row_group_idx]};

  using equality_fn_type    = insert_equality_functor<cudf::string_view>;
  using hash_fn_type        = insert_hash_functor<cudf::string_view>;
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  auto hash_set_ref = cuco::static_set_ref{cuco::empty_key<key_type>{empty_key_sentinel},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_thread,
                                           storage_ref};

  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page_data_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  auto const is_error_set = [&]() {
    return cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block>{*error}.load(
             cuda::std::memory_order_relaxed) != 0;
  };

  auto const value_offset = value_offsets[row_group_idx];

  // Decode with single warp thread until the value is found or we reach the end of the page
  if (warp.thread_rank() == 0) {
    auto buffer_offset  = int32_t{0};
    auto decoded_values = key_type{0};
    while (buffer_offset < page_data_size) {
      if (decoded_values > page.num_input_values or
          is_stream_overrun(buffer_offset, sizeof(int32_t))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode string length
      auto const string_length = static_cast<int32_t>(*(page_data + buffer_offset));
      buffer_offset += sizeof(int32_t);
      if (is_stream_overrun(buffer_offset, string_length)) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode cudf::string_view value
      decoded_data[value_offset + decoded_values] =
        cudf::string_view{reinterpret_cast<char const*>(page_data + buffer_offset),
                          static_cast<cudf::size_type>(string_length)};

      set_insert_ref.insert(static_cast<key_type>(value_offset + decoded_values));

      // Otherwise, keep going
      buffer_offset += string_length;
      decoded_values++;

      // Break if an error has been set
      if (is_error_set()) { break; }
    }
  }
}

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
build_fixed_width_dictionaries(PageInfo const* pages,
                               cudf::device_span<T> decoded_data,
                               bucket_type* const set_storage,
                               cudf::size_type const* set_offsets,
                               cudf::size_type const* value_offsets,
                               parquet::Type physical_type,
                               cudf::size_type num_dictionary_columns,
                               cudf::size_type dictionary_col_idx,
                               kernel_error::pointer error,
                               cudf::size_type flba_length = 0)
{
  namespace cg             = cooperative_groups;
  auto const group         = cg::this_thread_block();
  auto const row_group_idx = cg::this_grid().block_rank();
  auto const chunk_idx     = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page         = pages[chunk_idx];
  auto const& page_data    = page.page_data;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) { return; }

  auto const value_offset = value_offsets[row_group_idx];
  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage + set_offsets[row_group_idx]};

  using equality_fn_type = insert_equality_functor<T>;
  using hash_fn_type     = insert_hash_functor<T>;
  // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
  // benchmarks.
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  auto hash_set_ref = cuco::static_set_ref{cuco::empty_key{empty_key_sentinel},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_block,
                                           storage_ref};

  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page.uncompressed_page_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  auto const is_error_set = [&]() {
    return cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block>{*error}.load(
             cuda::std::memory_order_relaxed) != 0;
  };

  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    auto const insert_key = static_cast<key_type>(value_offset + value_idx);
    if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      // Parquet physical type must be fixed length so either INT96 or FIXED_LEN_BYTE_ARRAY
      switch (physical_type) {
        case parquet::Type::INT96: flba_length = int96_size; [[fallthrough]];
        case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
          // Check if we are overruning the data stream
          if (is_stream_overrun(value_idx * flba_length, flba_length)) {
            set_error(decode_error::DATA_STREAM_OVERRUN);
          }
          decoded_data[value_offset + value_idx] = cudf::string_view{
            reinterpret_cast<char const*>(page_data) + value_idx * flba_length, flba_length};
          set_insert_ref.insert(insert_key);
          break;
        }
        default: {
          // Parquet physical type is not fixed length so set the error code and break early
          set_error(decode_error::INVALID_DATA_TYPE);
        }
      }
    } else {
      // Check if we are overruning the data stream
      if (is_stream_overrun(value_idx * sizeof(T), sizeof(T))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
      }
      // Simply copy over the decoded value bytes from page data
      cuda::std::memcpy(
        &decoded_data[value_offset + value_idx], page_data + (value_idx * sizeof(T)), sizeof(T));
      set_insert_ref.insert(insert_key);
    }
    // Return early if an error has been set
    if (is_error_set()) { return; }
  }
}

__global__ void evaluate_some_string_literals(PageInfo const* pages,
                                              cudf::device_span<bool*> results,
                                              ast::generic_scalar_device_view const* scalars,
                                              ast::ast_operator const* operators,
                                              cudf::size_type total_num_scalars,
                                              cudf::size_type total_row_groups,
                                              cudf::size_type num_dictionary_columns,
                                              cudf::size_type dictionary_col_idx,
                                              kernel_error::pointer error)
{
  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const row_group_idx =
    (cg::this_grid().block_rank() * warp.meta_group_size()) + warp.meta_group_rank();

  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page_data_size == 0) {
    if (warp.thread_rank() == 0 and row_group_idx < total_row_groups) {
      for (auto i = 0; i < total_num_scalars; ++i) {
        results[i][row_group_idx] = operators[i] == ast::ast_operator::EQUAL;
      }
    }
    return;
  }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page_data_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  if (row_group_idx >= total_row_groups) { return; }

  // Decode with single warp thread until the value is found or we reach the end of the page
  if (warp.thread_rank() == 0) {
    for (auto i = 0; i < total_num_scalars; ++i) {
      results[i][row_group_idx] = false;
    }
    int32_t buffer_offset          = 0;
    cudf::size_type decoded_values = 0;
    while (buffer_offset < page_data_size) {
      // Check for errors
      if (decoded_values > page.num_input_values or
          is_stream_overrun(buffer_offset, sizeof(int32_t))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode string length
      auto const string_length = static_cast<int32_t>(*(page_data + buffer_offset));
      buffer_offset += sizeof(int32_t);
      if (is_stream_overrun(buffer_offset, string_length)) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode cudf::string_view value
      auto const decoded_value =
        cudf::string_view{reinterpret_cast<char const*>(page_data + buffer_offset),
                          static_cast<cudf::size_type>(string_length)};

      // If the decoded value is equal to the scalar value, set the result to true and return
      // early
      for (auto scalar_idx = 0; scalar_idx < total_num_scalars; ++scalar_idx) {
        if (decoded_value == scalars[scalar_idx].value<cudf::string_view>()) {
          results[scalar_idx][row_group_idx] = operators[scalar_idx] == ast::ast_operator::NOT_EQUAL
                                                 ? page.num_input_values == 1
                                                 : true;
        }
      }
      // Otherwise, keep going
      buffer_offset += string_length;
      decoded_values++;

      // Break if we have found all literals
      if (thrust::all_of(thrust::seq,
                         thrust::counting_iterator(0),
                         thrust::counting_iterator(total_num_scalars),
                         [&](auto scalar_idx) {
                           return operators[scalar_idx] == ast::ast_operator::NOT_EQUAL
                                    ? page.num_input_values > 1 or
                                        results[scalar_idx][row_group_idx]
                                    : results[scalar_idx][row_group_idx];
                         })) {
        break;
      }
    }
  }
}

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
evaluate_some_fixed_width_literals(PageInfo const* pages,
                                   cudf::device_span<bool*> results,
                                   ast::generic_scalar_device_view const* scalars,
                                   ast::ast_operator const* operators,
                                   parquet::Type physical_type,
                                   cudf::size_type total_num_scalars,
                                   cudf::size_type num_dictionary_columns,
                                   cudf::size_type dictionary_col_idx,
                                   kernel_error::pointer error,
                                   cudf::size_type flba_length = 0)
{
  namespace cg             = cooperative_groups;
  auto const group         = cg::this_thread_block();
  auto const row_group_idx = cg::this_grid().block_rank();
  auto const chunk_idx     = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page         = pages[chunk_idx];
  auto const& page_data    = page.page_data;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) {
    for (auto i = group.thread_rank(); i < total_num_scalars; i += group.num_threads()) {
      results[i][row_group_idx] = operators[i] == ast::ast_operator::EQUAL;
    }
    return;
  }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page.uncompressed_page_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  auto const is_error_set = [&]() {
    return cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block>{*error}.load(
             cuda::std::memory_order_relaxed) != 0;
  };

  for (auto i = group.thread_rank(); i < total_num_scalars; i += group.num_threads()) {
    results[i][row_group_idx] = false;
  }

  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    // Placeholder for the decoded value
    auto decoded_value = T{};

    if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      // Parquet physical type must be fixed length so either INT96 or FIXED_LEN_BYTE_ARRAY
      switch (physical_type) {
        case parquet::Type::INT96: flba_length = int96_size; [[fallthrough]];
        case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
          // Check if we are overruning the data stream
          if (is_stream_overrun(value_idx * flba_length, flba_length)) {
            set_error(decode_error::DATA_STREAM_OVERRUN);
            return;
          }
          decoded_value = cudf::string_view{
            reinterpret_cast<char const*>(page_data) + value_idx * flba_length, flba_length};
          break;
        }
        default: {
          // Parquet physical type is not fixed length so set the error code and break early
          set_error(decode_error::INVALID_DATA_TYPE);
          return;
        }
      }
    } else {
      // Check if we are overruning the data stream
      if (is_stream_overrun(value_idx * sizeof(T), sizeof(T))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        return;
      }
      // Simply copy over the decoded value bytes from page data
      cuda::std::memcpy(&decoded_value, page_data + (value_idx * sizeof(T)), sizeof(T));
    }

    // If the decoded value is equal to the scalar value, set the result to true and return early
    for (auto scalar_idx = 0; scalar_idx < total_num_scalars; ++scalar_idx) {
      if (decoded_value == scalars[scalar_idx].value<T>()) {
        results[scalar_idx][row_group_idx] =
          operators[scalar_idx] == ast::ast_operator::NOT_EQUAL ? page.num_input_values == 1 : true;
      }
    }

    // If we have already found matches or error, return early
    if (is_error_set() or
        thrust::all_of(thrust::seq,
                       thrust::counting_iterator(0),
                       thrust::counting_iterator(total_num_scalars),
                       [&](auto scalar_idx) {
                         return operators[scalar_idx] == ast::ast_operator::NOT_EQUAL
                                  ? page.num_input_values > 1 or results[scalar_idx][row_group_idx]
                                  : results[scalar_idx][row_group_idx];
                       })) {
      return;
    }
  }
}

/**
 * @brief Converts dictionary membership results (for each column chunk) to a device column.
 */
struct dictionary_caster {
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks;
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages;
  size_t total_row_groups;
  parquet::Type physical_type;
  cudf::size_type type_length;
  cudf::size_type num_dictionary_columns;
  cudf::size_type dictionary_col_idx;
  rmm::cuda_stream_view stream;

  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> build_columns(
    cudf::host_span<rmm::device_buffer> results_buffers)
  {
    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.reserve(results_buffers.size());
    std::transform(results_buffers.begin(),
                   results_buffers.end(),
                   std::back_inserter(columns),
                   [&](auto& result_buffer) {
                     return std::make_unique<cudf::column>(
                       cudf::data_type{cudf::type_id::BOOL8},
                       static_cast<cudf::size_type>(total_row_groups),
                       std::move(result_buffer),
                       rmm::device_buffer{},
                       0);
                   });
    return columns;
  }

  template <typename T>
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                   std::vector<std::unique_ptr<cudf::column>>>
  evaluate_many_literals(cudf::host_span<ast::literal* const> literals,
                         cudf::host_span<ast::ast_operator const> operators)
  {
    auto host_set_offsets   = std::vector<cudf::size_type>{};
    auto host_value_offsets = std::vector<cudf::size_type>{};
    host_set_offsets.reserve(total_row_groups + 1);
    host_value_offsets.reserve(total_row_groups + 1);
    host_set_offsets.emplace_back(0);
    host_value_offsets.emplace_back(0);
    std::for_each(
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator(total_row_groups),
      [&](auto row_group_idx) {
        auto const chunk_idx        = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
        auto const num_input_values = pages[chunk_idx].num_input_values;
        host_value_offsets.emplace_back(host_value_offsets.back() + num_input_values);
        host_set_offsets.emplace_back(host_set_offsets.back() +
                                      static_cast<cudf::size_type>(compute_hash_table_size(
                                        num_input_values, occupancy_factor)));
      });

    auto const set_offsets = cudf::detail::make_device_uvector_async(
      host_set_offsets, stream, cudf::get_current_device_resource_ref());
    auto const value_offsets = cudf::detail::make_device_uvector_async(
      host_value_offsets, stream, cudf::get_current_device_resource_ref());

    auto const total_bucket_storage_size = static_cast<size_t>(host_set_offsets.back());
    auto const total_num_values          = static_cast<size_t>(host_value_offsets.back());
    auto const total_num_literals        = static_cast<cudf::size_type>(literals.size());

    // Create a single bulk storage used by all sub-dictionaries
    auto set_storage = storage_type{
      total_bucket_storage_size,
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream}};

    // Initialize storage with the empty key sentinel
    set_storage.initialize_async(empty_key_sentinel, {stream.value()});

    rmm::device_uvector<T> decoded_data{
      total_num_values, stream, cudf::get_current_device_resource_ref()};
    kernel_error error_code(stream);

    std::vector<ast::generic_scalar_device_view> host_scalars;
    host_scalars.reserve(total_num_literals);
    std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
      host_scalars.push_back(literal->get_value());
    });

    auto const scalars = cudf::detail::make_device_uvector_async(
      host_scalars, stream, cudf::get_current_device_resource_ref());
    auto const d_operators = cudf::detail::make_device_uvector_async(
      operators, stream, cudf::get_current_device_resource_ref());

    auto query_block_size = [&]() {
      auto query_block_size = std::max<cudf::size_type>(cudf::detail::warp_size, total_row_groups);
      query_block_size      = cudf::size_type{1} << (31 - __builtin_clz(query_block_size));
      return std::min<cudf::size_type>(query_block_size, 128);
    }();

    std::vector<rmm::device_buffer> results_buffers(total_num_literals);
    std::vector<bool*> host_results_ptrs(total_num_literals);
    std::for_each(
      thrust::counting_iterator(0), thrust::counting_iterator(total_num_literals), [&](auto i) {
        results_buffers[i] =
          rmm::device_buffer(total_row_groups, stream, cudf::get_current_device_resource_ref());
        host_results_ptrs[i] = static_cast<bool*>(results_buffers[i].data());
      });
    auto results_ptrs = cudf::detail::make_device_uvector_async(
      host_results_ptrs, stream, cudf::get_current_device_resource_ref());

    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      build_fixed_width_dictionaries<T>
        <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                     decoded_data,
                                                                     set_storage.data(),
                                                                     set_offsets.data(),
                                                                     value_offsets.data(),
                                                                     physical_type,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());

      // Check if there are any errors in data decoding
      if (auto const error = error_code.value_sync(stream); error != 0) {
        CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
      }

      query_dictionaries<T>
        <<<total_num_literals, query_block_size, 0, stream.value()>>>(decoded_data,
                                                                      results_ptrs,
                                                                      scalars.data(),
                                                                      d_operators.data(),
                                                                      set_storage.data(),
                                                                      set_offsets.data(),
                                                                      value_offsets.data(),
                                                                      total_row_groups,
                                                                      physical_type);
    } else {
      if (physical_type == parquet::Type::INT96 or
          physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
        // Get flba length from the first column chunk of this column
        auto const flba_length = physical_type == parquet::Type::INT96
                                   ? int96_size
                                   : chunks[dictionary_col_idx].type_length;
        // Check if the fixed width literal is in the dictionaries
        build_fixed_width_dictionaries<T>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                       decoded_data,
                                                                       set_storage.data(),
                                                                       set_offsets.data(),
                                                                       value_offsets.data(),
                                                                       physical_type,
                                                                       num_dictionary_columns,
                                                                       dictionary_col_idx,
                                                                       error_code.data(),
                                                                       flba_length);
        // Check if there are any errors in data decoding
        if (auto const error = error_code.value_sync(stream); error != 0) {
          CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
        }

        query_dictionaries<T>
          <<<total_num_literals, query_block_size, 0, stream.value()>>>(decoded_data,
                                                                        results_ptrs,
                                                                        scalars.data(),
                                                                        d_operators.data(),
                                                                        set_storage.data(),
                                                                        set_offsets.data(),
                                                                        value_offsets.data(),
                                                                        total_row_groups,
                                                                        physical_type);
      } else {
        static_assert(decode_block_size % cudf::detail::warp_size == 0,
                      "decode_block_size must be a multiple of warp_size");
        size_t const warps_per_block = decode_block_size / cudf::detail::warp_size;
        auto const num_blocks =
          cudf::util::div_rounding_up_safe<size_t>(total_row_groups, warps_per_block);

        build_string_dictionaries<<<num_blocks, decode_block_size, 0, stream.value()>>>(
          pages.device_begin(),
          decoded_data,
          set_storage.data(),
          set_offsets.data(),
          value_offsets.data(),
          total_row_groups,
          num_dictionary_columns,
          dictionary_col_idx,
          error_code.data());

        // Check if there are any errors in data decoding
        if (auto const error = error_code.value_sync(stream); error != 0) {
          CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
        }

        query_dictionaries<cudf::string_view>
          <<<total_num_literals, query_block_size, 0, stream.value()>>>(decoded_data,
                                                                        results_ptrs,
                                                                        scalars.data(),
                                                                        d_operators.data(),
                                                                        set_storage.data(),
                                                                        set_offsets.data(),
                                                                        value_offsets.data(),
                                                                        total_row_groups,
                                                                        physical_type);
      }
    }

    return build_columns(results_buffers);
  }

  template <typename T>
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                   std::vector<std::unique_ptr<cudf::column>>>
  evaluate_some_literals(cudf::host_span<ast::literal* const> literals,
                         cudf::host_span<ast::ast_operator const> operators)
  {
    std::vector<ast::generic_scalar_device_view> host_scalars;
    host_scalars.reserve(literals.size());
    std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
      host_scalars.push_back(literal->get_value());
    });

    auto const scalars = cudf::detail::make_device_uvector_async(
      host_scalars, stream, cudf::get_current_device_resource_ref());
    auto const d_operators = cudf::detail::make_device_uvector_async(
      operators, stream, cudf::get_current_device_resource_ref());
    auto const total_num_scalars  = static_cast<cudf::size_type>(scalars.size());
    auto const total_num_literals = static_cast<cudf::size_type>(literals.size());

    std::vector<rmm::device_buffer> results_buffers(total_num_literals);
    std::vector<bool*> host_results_ptrs(total_num_literals);
    std::for_each(
      thrust::counting_iterator(0), thrust::counting_iterator(total_num_literals), [&](auto i) {
        results_buffers[i] =
          rmm::device_buffer(total_row_groups, stream, cudf::get_current_device_resource_ref());
        host_results_ptrs[i] = static_cast<bool*>(results_buffers[i].data());
      });

    auto results_ptrs = cudf::detail::make_device_uvector_async(
      host_results_ptrs, stream, cudf::get_current_device_resource_ref());

    kernel_error error_code(stream);

    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      evaluate_some_fixed_width_literals<T>
        <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                     results_ptrs,
                                                                     scalars.data(),
                                                                     d_operators.data(),
                                                                     physical_type,
                                                                     total_num_scalars,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());
    } else {
      if (physical_type == parquet::Type::INT96 or
          physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
        auto const flba_length = physical_type == parquet::Type::INT96
                                   ? int96_size
                                   : chunks[dictionary_col_idx].type_length;
        evaluate_some_fixed_width_literals<T>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                       results_ptrs,
                                                                       scalars.data(),
                                                                       d_operators.data(),
                                                                       physical_type,
                                                                       total_num_scalars,
                                                                       num_dictionary_columns,
                                                                       dictionary_col_idx,
                                                                       error_code.data(),
                                                                       flba_length);
      } else {
        static_assert(decode_block_size % cudf::detail::warp_size == 0,
                      "decode_block_size must be a multiple of warp_size");
        size_t const warps_per_block = decode_block_size / cudf::detail::warp_size;
        auto const num_blocks =
          cudf::util::div_rounding_up_safe<size_t>(total_row_groups, warps_per_block);

        evaluate_some_string_literals<<<num_blocks, decode_block_size, 0, stream.value()>>>(
          pages.device_begin(),
          results_ptrs,
          scalars.data(),
          d_operators.data(),
          total_num_scalars,
          total_row_groups,
          num_dictionary_columns,
          dictionary_col_idx,
          error_code.data());
      }
    }

    if (auto const error = error_code.value_sync(stream); error != 0) {
      CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
    }

    return build_columns(results_buffers);
  }

  template <typename T>
  std::vector<std::unique_ptr<cudf::column>> operator()(
    cudf::data_type dtype,
    cudf::host_span<ast::literal* const> literals,
    cudf::host_span<ast::ast_operator const> operators)
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (cuda::std::is_same_v<T, bool> or
                  (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>)) {
      CUDF_FAIL("Dictionaries do not support boolean or compound types");
    } else {
      // Make sure all literals have the same type as the predicate column
      std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
        // Check if the literal has the same type as the predicate column
        CUDF_EXPECTS(
          dtype == literal->get_data_type() and
            cudf::have_same_types(
              cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
              cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
          "Mismatched predicate column and literal types");
      });

      // If there is only one literal, just evaluate expression while decoding dictionary data
      if (literals.size() <= max_inline_literals) {
        return evaluate_some_literals<T>(literals, operators);
      } else {
        // Else, decode dictionaries to `cudf::static_set`s and evaluate all expressions
        return evaluate_many_literals<T>(literals, operators);
      }
    }
  }
};

/**
 * @brief Converts AST expression to dictionary membership (DictionaryAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class dictionary_expression_converter : public equality_literals_collector {
 public:
  dictionary_expression_converter(ast::expression const& expr,
                                  size_type num_input_columns,
                                  cudf::host_span<std::vector<ast::literal*> const> literals)
    : _literals{literals}
  {
    // Set the num columns
    _num_input_columns = num_input_columns;

    // Compute and store columns literals offsets
    _col_literals_offsets.reserve(_num_input_columns + 1);
    _col_literals_offsets.emplace_back(0);

    std::transform(literals.begin(),
                   literals.end(),
                   std::back_inserter(_col_literals_offsets),
                   [&](auto const& col_literal_map) {
                     return _col_literals_offsets.back() +
                            static_cast<cudf::size_type>(col_literal_map.size());
                   });

    // Add this visitor
    expr.accept(*this);
  }

  /**
   * @brief Delete equality literals getter as it's not needed in the derived class
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_equality_literals() && = delete;

  // Bring all overloads of `visit` from equality_predicate_collector into scope
  using equality_literals_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    using cudf::ast::ast_operator;
    auto const operands = expr.get_operands();
    auto const op       = expr.get_operator();

    if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
      // First operand should be column reference, second should be literal.
      CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                   "Only binary operations are supported on column reference");
      CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      v->accept(*this);

      if (op == ast_operator::EQUAL or op == ast::ast_operator::NOT_EQUAL) {
        // Search the literal in this input column's equality literals list and add to
        // the offset.
        auto const col_idx            = v->get_column_index();
        auto const& equality_literals = _literals[col_idx];
        auto col_literal_offset       = _col_literals_offsets[col_idx];
        auto const literal_iter       = std::find(equality_literals.cbegin(),
                                            equality_literals.cend(),
                                            dynamic_cast<ast::literal const*>(&operands[1].get()));
        CUDF_EXPECTS(literal_iter != equality_literals.end(), "Could not find the literal ptr");
        col_literal_offset += std::distance(equality_literals.cbegin(), literal_iter);

        // Evaluate boolean is_true(value) expression as NOT(NOT(value))
        auto const& value = _dictionary_expr.push(ast::column_reference{col_literal_offset});

        auto const& not_in_dictionary =
          _dictionary_expr.push(ast::operation{ast_operator::NOT, value});

        if (op == ast_operator::EQUAL) {
          _dictionary_expr.push(ast::operation{ast_operator::NOT, not_in_dictionary});
        }
      }
      // For all other expressions, push an always true expression
      else {
        _dictionary_expr.push(
          ast::operation{ast_operator::NOT,
                         _dictionary_expr.push(ast::operation{ast_operator::NOT, _always_true})});
      }
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2) {
        _dictionary_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
      } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
        _dictionary_expr.push(ast::operation{op, new_operands.front()});
      }
    }
    return _dictionary_expr.back();
  }

  /**
   * @brief Returns the AST to apply on dictionary membership.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_dictionary_expr() const
  {
    return _dictionary_expr.back();
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  cudf::host_span<std::vector<ast::literal*> const> _literals;
  ast::tree _dictionary_expr;
  cudf::numeric_scalar<bool> _always_true_scalar{true};
  ast::literal const _always_true{_always_true_scalar};
};

}  // namespace

std::optional<std::vector<std::vector<cudf::size_type>>>
aggregate_reader_metadata::apply_dictionary_filter(
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
  host_span<std::vector<size_type> const> input_row_group_indices,
  host_span<std::vector<ast::literal*> const> literals,
  cudf::host_span<std::vector<ast::ast_operator> const> operators,
  std::size_t total_row_groups,
  cudf::host_span<data_type const> output_dtypes,
  cudf::host_span<int const> dictionary_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());
  // Number of dictionary columns
  auto const num_dictionary_columns = static_cast<cudf::size_type>(dictionary_col_schemas.size());

  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(input_row_group_indices, dictionary_col_schemas);

  // Converts dictionary membership for (in)equality predicate columns to a table
  // containing a column for each `col[i] == literal` or `col[i] != literal` predicate to be
  // evaluated. The table contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<cudf::column>> dictionary_membership_columns;
  cudf::size_type dictionary_col_idx = 0;
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(output_dtypes.size()),
    [&](auto input_col_idx) {
      auto const& dtype = output_dtypes[input_col_idx];

      // Skip if no equality literals for this column
      if (literals[input_col_idx].empty()) { return; }

      // Skip if non-comparable (compound) type except string
      if (cudf::is_compound(dtype) and dtype.id() != cudf::type_id::STRING) { return; }

      auto const type_length = chunks[dictionary_col_idx].type_length;

      // Create a bloom filter query table caster
      dictionary_caster const dictionary_col{chunks,
                                             pages,
                                             total_row_groups,
                                             parquet_types[dictionary_col_idx],
                                             type_length,
                                             num_dictionary_columns,
                                             dictionary_col_idx,
                                             stream};

      // Add a column for all literals associated with an equality column
      auto dict_columns = cudf::type_dispatcher<dispatch_storage_type>(
        dtype, dictionary_col, dtype, literals[input_col_idx], operators[input_col_idx]);

      dictionary_membership_columns.insert(dictionary_membership_columns.end(),
                                           std::make_move_iterator(dict_columns.begin()),
                                           std::make_move_iterator(dict_columns.end()));

      dictionary_col_idx++;
    });

  // Create a table from columns
  auto const dictionary_membership_table = cudf::table(std::move(dictionary_membership_columns));

  // Convert AST to DictionaryAST expression with reference to dictionary membership
  // in above `dictionary_membership_table`
  dictionary_expression_converter dictionary_expr{filter.get(), num_input_columns, literals};

  // Filter dictionary membership table with the DictionaryAST expression and collect
  // filtered row group indices
  return parquet::detail::collect_filtered_row_group_indices(dictionary_membership_table,
                                                             dictionary_expr.get_dictionary_expr(),
                                                             input_row_group_indices,
                                                             stream);
}

dictionary_literals_collector::dictionary_literals_collector(ast::expression const& expr,
                                                             cudf::size_type num_input_columns)
{
  _num_input_columns = num_input_columns;
  _literals.resize(num_input_columns);
  _operators.resize(num_input_columns);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> dictionary_literals_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();

  if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
    // First operand should be column reference, second should be literal.
    CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                 "Only binary operations are supported on column reference");
    auto const literal_ptr = dynamic_cast<ast::literal const*>(&operands[1].get());
    CUDF_EXPECTS(literal_ptr != nullptr,
                 "Second operand of binary operation with column reference must be a literal");
    v->accept(*this);

    // Push to the corresponding column's literals and operators list iff EQUAL or NOT_EQUAL
    // operator is seen
    if (op == ast_operator::EQUAL or op == ast::ast_operator::NOT_EQUAL) {
      auto const col_idx = v->get_column_index();
      _literals[col_idx].emplace_back(const_cast<ast::literal*>(literal_ptr));
      _operators[col_idx].emplace_back(op);
    }
  } else {
    // Just visit the operands and ignore any output
    std::ignore = visit_operands(operands);
  }

  return expr;
}

std::pair<std::vector<std::vector<ast::literal*>>, std::vector<std::vector<ast::ast_operator>>>
dictionary_literals_collector::get_literals_and_operators() &&
{
  return {std::move(_literals), std::move(_operators)};
}

}  // namespace cudf::io::parquet::experimental::detail
