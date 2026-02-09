/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "orc.hpp"

#include <cstdint>

/**
 * @file orc_field_reader.hpp
 * @brief Functors to encapsulate common functionality required to implement
 * protobuf_reader::read(...) functions
 */

namespace cudf::io::orc::detail {

/**
 * @brief Functor to run an operator for a specified field.
 *
 * The purpose of this functor is to replace a switch case. If the field in
 * the argument is equal to the field specified in any element of the tuple
 * of operators then it is run with the byte stream and field type arguments.
 *
 * If the field does not match any of the functors then skip_struct_field is
 * called by the protobuf_reader
 */
template <int index>
struct function_switch {
  template <typename... Operator>
  static inline void run(protobuf_reader* pbr,
                         uint8_t const* end,
                         int const& encoded_field_number,
                         std::tuple<Operator...>& ops)
  {
    if (encoded_field_number == std::get<index>(ops).encoded_field_number) {
      std::get<index>(ops)(pbr, end);
    } else {
      function_switch<index - 1>::run(pbr, end, encoded_field_number, ops);
    }
  }
};

template <>
struct function_switch<0> {
  template <typename... Operator>
  static inline void run(protobuf_reader* pbr,
                         uint8_t const* end,
                         int const& encoded_field_number,
                         std::tuple<Operator...>& ops)
  {
    if (encoded_field_number == std::get<0>(ops).encoded_field_number) {
      std::get<0>(ops)(pbr, end);
    } else {
      pbr->skip_struct_field(encoded_field_number & 7);
    }
  }
};

/**
 * @brief Function to implement protobuf_reader::read based on the tuple of functors provided.
 *
 * Bytes are read from the internal metadata stream and field types are matched up against user
 * supplied reading functors. If they match then the corresponding values are written to references
 * pointed to by the functors.
 */
template <typename T, typename... Operator>
inline void protobuf_reader::function_builder(T& s, size_t maxlen, std::tuple<Operator...>& op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  auto* const end     = std::min(m_cur + maxlen, m_end);
  while (m_cur < end) {
    auto const field = get<uint32_t>();
    function_switch<index>::run(this, end, field, op);
  }
  CUDF_EXPECTS(m_cur <= end, "Current pointer to metadata stream is out of bounds");
}

}  // namespace cudf::io::orc::detail
