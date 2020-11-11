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
 *
 * Returns the output of the selected operator or false if no operator
 * matched the field value
 */
template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline bool run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &field,
                         std::tuple<Operator...> &ops)
  {
    if (field == std::get<index>(ops).field) {
      return std::get<index>(ops)(pbr, end);
    } else {
      return FunctionSwitchImpl<index - 1>::run(pbr, end, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline bool run(ProtobufReader *pbr,
                         const uint8_t *end,
                         const int &field,
                         std::tuple<Operator...> &ops)
  {
    if (field == std::get<0>(ops).field) {
      return std::get<0>(ops)(pbr, end);
    } else {
      pbr->skip_struct_field(field & 7);
      return false;
    }
  }
};

/**
 * @brief Function to ascertain the return value of ProtobufReader::read
 * function
 *
 * @return Returns 'false' if current pointer to metadata stream is out of
 * bounds
 */
template <typename T>
inline bool ProtobufReader::function_builder_return(T &s, const uint8_t *end)
{
  return m_cur <= end;
}

/**
 * @brief Function to ascertain the return value of
 * `ProtobufReader::read(FileFooter*, ...)` function
 *
 * @return Returns 'false' if current pointer to metadata stream is out of
 * bounds or if the initialization of the parent_idx field of FileFooter
 * is not done correctly
 */
template <>
inline bool ProtobufReader::function_builder_return<FileFooter>(FileFooter &s, const uint8_t *end)
{
  return (m_cur <= end) && InitSchema(s);
}

/**
 * @brief Function to implement ProtobufReader::read based on the tuple of
 * functors provided
 *
 * Bytes are read from the internal metadata stream and field types are
 * matched up against user supplied reading functors. If they match then the
 * corresponding values are written to references pointed to by the functors.
 *
 * @return Returns 'false' if an unexpected field is encountered while reading.
 * Otherwise 'true' is returned.
 */
template <typename T, typename... Operator>
inline bool ProtobufReader::function_builder(T &s, size_t maxlen, std::tuple<Operator...> &op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  const uint8_t *end  = std::min(m_cur + maxlen, m_end);
  while (m_cur < end) {
    int field          = get_u32();
    bool exit_function = FunctionSwitchImpl<index>::run(this, end, field, op);
    if (exit_function) { return false; }
  }
  return function_builder_return(s, end);
}

/**
 * @brief Functor to set value to 32 bit integer read from metadata stream
 *
 * Returns 'false'
 */
struct ProtobufReader::FieldInt32 {
  int field;
  int32_t &value;

  FieldInt32(int f, int32_t &v) : field((f * 8) + PB_TYPE_VARINT), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    value = pbr->get_i32();
    return false;
  }
};

/**
 * @brief Functor to set value to 32 bit unsigned integer read from metadata
 * stream
 *
 * Returns 'false'
 */
struct ProtobufReader::FieldUInt32 {
  int field;
  uint32_t &value;

  FieldUInt32(int f, uint32_t &v) : field((f * 8) + PB_TYPE_VARINT), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    value = pbr->get_u32();
    return false;
  }
};

/**
 * @brief Functor to set value to 64 bit integer read from metadata stream
 *
 * Returns 'false'
 */
struct ProtobufReader::FieldInt64 {
  int field;
  int64_t &value;

  FieldInt64(int f, int64_t &v) : field((f * 8) + PB_TYPE_VARINT), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    value = pbr->get_i64();
    return false;
  }
};

/**
 * @brief Functor to set value to 64 bit unsigned integer read from metadata
 * stream
 *
 * Returns 'false'
 */
struct ProtobufReader::FieldUInt64 {
  int field;
  uint64_t &value;

  FieldUInt64(int f, uint64_t &v) : field((f * 8) + PB_TYPE_VARINT), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    value = pbr->get_u64();
    return false;
  }
};

/**
 * @brief Functor to set value to enum read from metadata stream
 *
 * Returns 'false'
 */
template <typename Enum>
struct ProtobufReader::FieldEnum {
  int field;
  Enum &value;

  FieldEnum(int f, Enum &v) : field((f * 8) + PB_TYPE_VARINT), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    value = static_cast<Enum>(pbr->get_u32());
    return false;
  }
};

/**
 * @brief Functor to append a 32 bit integer to a vector of integers
 * read from metadata stream
 *
 * Returns 'false'
 */
struct ProtobufReader::FieldPackedUInt32 {
  int field;
  std::vector<uint32_t> &value;

  FieldPackedUInt32(int f, std::vector<uint32_t> &v) : field((f * 8) + PB_TYPE_FIXEDLEN), value(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t len             = pbr->get_u32();
    const uint8_t *field_end = std::min(pbr->m_cur + len, end);
    while (pbr->m_cur < field_end) value.push_back(pbr->get_u32());
    return false;
  }
};

/**
 * @brief Functor to set value to string read from metadata stream
 *
 * Returns 'true' if the length of the string exceeds bounds of the
 * metadata stream
 */
struct ProtobufReader::FieldString {
  int field;
  std::string &value;

  FieldString(int f, std::string &v) : field((f * 8) + PB_TYPE_FIXEDLEN), value(v) {}

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    value.assign((const char *)(pbr->m_cur), n);
    pbr->m_cur += n;
    return false;
  }
};

/**
 * @brief Functor to append a string read from metadata stream
 * to a vector of strings
 *
 * Returns 'true' if the length of the string exceeds bounds of the
 * metadata stream
 */
struct ProtobufReader::FieldRepeatedString {
  int field;
  std::vector<std::string> &value;

  FieldRepeatedString(int f, std::vector<std::string> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), value(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    value.resize(value.size() + 1);
    value.back().assign((const char *)(pbr->m_cur), n);
    pbr->m_cur += n;
    return false;
  }
};

/**
 * @brief Functor to append an enum read from metadata stream
 * to a vector of enums
 *
 * Returns 'true' if the maximum length read by the stream could
 * cause out of bounds read of the buffer or if the process of
 * reading the struct fails
 */
template <typename Enum>
struct ProtobufReader::FieldRepeatedStructFunctor {
  int field;
  std::vector<Enum> &value;

  FieldRepeatedStructFunctor(int f, std::vector<Enum> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), value(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    value.resize(value.size() + 1);
    if (!pbr->read(value.back(), n)) return true;
    return false;
  }
};

/**
 * @brief Functor to append an enum read from metadata stream
 * to a vector of enums
 *
 * Returns 'true' if the maximum length read by the stream could
 * cause out of bounds read of the buffer
 */
template <typename Enum>
struct ProtobufReader::FieldRepeatedStructBlobFunctor {
  int field;
  std::vector<Enum> &value;

  FieldRepeatedStructBlobFunctor(int f, std::vector<Enum> &v)
    : field((f * 8) + PB_TYPE_FIXEDLEN), value(v)
  {
  }

  inline bool operator()(ProtobufReader *pbr, const uint8_t *end)
  {
    uint32_t n = pbr->get_u32();
    if (n > (size_t)(end - pbr->m_cur)) return true;
    value.resize(value.size() + 1);
    value.back().assign(pbr->m_cur, pbr->m_cur + n);
    pbr->m_cur += n;
    return false;
  }
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
