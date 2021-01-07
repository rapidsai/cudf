/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <io/orc/orc.h>
#include <string>

/**
 * @file orc_field_reader.hpp
 * @brief Functors to encapsulate common functionality required to implement
 * ProtobufWriter::read(...) functions
 */

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Functor to run an operator for a specified field.
 *
 * The purpose of this functor is to replace a switch case. If the field in
 * the argument is equal to the field specified in any element of the tuple
 * of operators then it is run with the byte stream and field type arguments.
 *
 * If the field does not match any of the functors then skip_struct_field is
 * called by the ProtobufReader
 */
template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline void run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &encoded_field_number,
                         std::tuple<Operator...> &ops)
  {
    if (encoded_field_number == std::get<index>(ops).encoded_field_number) {
      std::get<index>(ops)(pbr, end);
    } else {
      FunctionSwitchImpl<index - 1>::run(pbr, end, encoded_field_number, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline void run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &encoded_field_number,
                         std::tuple<Operator...> &ops)
  {
    if (encoded_field_number == std::get<0>(ops).encoded_field_number) {
      std::get<0>(ops)(pbr, end);
    } else {
      pbr->skip_struct_field(encoded_field_number & 7);
    }
  }
};

/**
 * @brief Function to implement ProtobufReader::read based on the tuple of functors provided.
 *
 * Bytes are read from the internal metadata stream and field types are matched up against user
 * supplied reading functors. If they match then the corresponding values are written to references
 * pointed to by the functors.
 */
template <typename T, typename... Operator>
inline void ProtobufReader::function_builder(T &s, size_t maxlen, std::tuple<Operator...> &op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  auto *const end     = std::min(m_cur + maxlen, m_end);
  while (m_cur < end) {
    auto const field = get<uint32_t>();
    FunctionSwitchImpl<index>::run(this, end, field, op);
  }
  CUDF_EXPECTS(m_cur <= end, "Current pointer to metadata stream is out of bounds");
}

}  // namespace orc
}  // namespace io
}  // namespace cudf
