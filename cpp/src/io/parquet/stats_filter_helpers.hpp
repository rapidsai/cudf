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

#pragma once

#include "io/parquet/parquet_common.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <numeric>
#include <string>

namespace cudf::io::parquet::detail {

/**
 * @brief Base utilities for converting and casting stats values
 *
 * Derived classes handle row group or page-level statistics as needed.
 *
 */
class stats_caster_base {
 protected:
  template <typename ToType, typename FromType>
  static inline ToType targetType(FromType const value)
  {
    if constexpr (cudf::is_timestamp<ToType>()) {
      return static_cast<ToType>(
        typename ToType::duration{static_cast<typename ToType::rep>(value)});
    } else if constexpr (std::is_same_v<ToType, string_view>) {
      return ToType{nullptr, 0};
    } else {
      return static_cast<ToType>(value);
    }
  }

  // uses storage type as T
  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>() or cudf::is_nested<T>())>
  static inline T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    CUDF_FAIL("unsupported type for stats casting");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_boolean<T>())>
  static inline T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    CUDF_EXPECTS(type == BOOLEAN, "Invalid type and stats combination");
    return stats_caster_base::targetType<T>(*reinterpret_cast<bool const*>(stats_val));
  }

  // integral but not boolean, and fixed_point, and chrono.
  template <typename T,
            CUDF_ENABLE_IF((cudf::is_integral<T>() and !cudf::is_boolean<T>()) or
                           cudf::is_fixed_point<T>() or cudf::is_chrono<T>())>
  static inline T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case INT32:
        return stats_caster_base::targetType<T>(*reinterpret_cast<int32_t const*>(stats_val));
      case INT64:
        return stats_caster_base::targetType<T>(*reinterpret_cast<int64_t const*>(stats_val));
      case INT96:  // Deprecated in parquet specification
        return stats_caster_base::targetType<T>(
          static_cast<__int128_t>(reinterpret_cast<int64_t const*>(stats_val)[0]) << 32 |
          reinterpret_cast<int32_t const*>(stats_val)[2]);
      case BYTE_ARRAY: [[fallthrough]];
      case FIXED_LEN_BYTE_ARRAY:
        if (stats_size == sizeof(T)) {
          // if type size == length of stats_val. then typecast and return.
          if constexpr (cudf::is_chrono<T>()) {
            return stats_caster_base::targetType<T>(
              *reinterpret_cast<typename T::rep const*>(stats_val));
          } else {
            return stats_caster_base::targetType<T>(*reinterpret_cast<T const*>(stats_val));
          }
        }
        // unsupported type
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_floating_point<T>())>
  static inline T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case FLOAT:
        return stats_caster_base::targetType<T>(*reinterpret_cast<float const*>(stats_val));
      case DOUBLE:
        return stats_caster_base::targetType<T>(*reinterpret_cast<double const*>(stats_val));
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, string_view>)>
  static inline T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case BYTE_ARRAY: [[fallthrough]];
      case FIXED_LEN_BYTE_ARRAY:
        return string_view(reinterpret_cast<char const*>(stats_val), stats_size);
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  /**
   * @brief Local struct to hold host columns during stats based filtering
   *
   * @tparam T Type of the column
   */
  template <typename T>
  struct host_column {
    // using thrust::host_vector because std::vector<bool> uses bitmap instead of byte per bool.
    cudf::detail::host_vector<T> val;
    std::vector<bitmask_type> null_mask;
    cudf::size_type null_count = 0;

    host_column(size_type total_row_groups, rmm::cuda_stream_view stream)
      : val{cudf::detail::make_host_vector<T>(total_row_groups, stream)},
        null_mask(cudf::util::div_rounding_up_safe<cudf::size_type>(
                    cudf::bitmask_allocation_size_bytes(total_row_groups), sizeof(bitmask_type)),
                  ~bitmask_type{0})
    {
    }

    void inline set_index(size_type index,
                          std::optional<std::vector<uint8_t>> const& binary_value,
                          Type const type)
    {
      if (binary_value.has_value()) {
        val[index] = stats_caster_base::convert<T>(
          binary_value.value().data(), binary_value.value().size(), type);
      }
      if (not binary_value.has_value()) {
        clear_bit_unsafe(null_mask.data(), index);
        null_count++;
      }
    }
    static inline std::tuple<rmm::device_uvector<char>, rmm::device_uvector<size_type>>
    make_strings_children(host_span<string_view> host_strings,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
    {
      auto const total_char_count =
        std::accumulate(host_strings.begin(), host_strings.end(), 0, [](auto sum, auto const& str) {
          return sum + str.size_bytes();
        });
      auto chars = cudf::detail::make_empty_host_vector<char>(total_char_count, stream);
      auto offsets =
        cudf::detail::make_empty_host_vector<cudf::size_type>(host_strings.size() + 1, stream);
      offsets.push_back(0);
      for (auto const& str : host_strings) {
        auto tmp =
          str.empty() ? std::string_view{} : std::string_view(str.data(), str.size_bytes());
        chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
        offsets.push_back(offsets.back() + tmp.length());
      }
      auto d_chars   = cudf::detail::make_device_uvector_async(chars, stream, mr);
      auto d_offsets = cudf::detail::make_device_uvector_sync(offsets, stream, mr);
      return std::tuple{std::move(d_chars), std::move(d_offsets)};
    }

    std::unique_ptr<column> inline to_device(cudf::data_type dtype,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
    {
      if constexpr (std::is_same_v<T, string_view>) {
        auto [d_chars, d_offsets] = make_strings_children(val, stream, mr);
        return cudf::make_strings_column(
          val.size(),
          std::make_unique<column>(std::move(d_offsets), rmm::device_buffer{}, 0),
          d_chars.release(),
          null_count,
          rmm::device_buffer{
            null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr});
      }
      return std::make_unique<column>(
        dtype,
        val.size(),
        cudf::detail::make_device_uvector_async(val, stream, mr).release(),
        rmm::device_buffer{
          null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr},
        null_count);
    }
  };
};

/**
 * @brief Converts AST expression to StatsAST for comparing with column statistics
 *
 * This is used in row group filtering based on predicate.
 * statistics min value of a column is referenced by column_index*2
 * statistics max value of a column is referenced by column_index*2+1
 */
class stats_expression_converter : public ast::detail::expression_transformer {
 public:
  stats_expression_converter(ast::expression const& expr, size_type num_columns);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the AST to apply on Column chunk statistics.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_stats_expr() const;

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  ast::tree _stats_expr;
  size_type _num_columns;
};

}  // namespace cudf::io::parquet::detail
