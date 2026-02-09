/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "orc.hpp"

#include <numeric>
#include <string>

/**
 * @file orc_field_writer.hpp
 * @brief Struct to encapsulate common functionality required to implement
 * `protobuf_writer::write(...)` functions
 */

namespace cudf::io::orc::detail {

struct protobuf_writer::protobuf_field_writer {
  int struct_size{0};
  protobuf_writer* p;

  protobuf_field_writer(protobuf_writer* pbw) : p(pbw) {}

  /**
   * @brief Function to write a unsigned integer to the internal buffer
   */
  template <typename T>
  void field_uint(int field, T const& value)
  {
    struct_size += p->put_uint(encode_field_number<T>(field));
    struct_size += p->put_uint(static_cast<uint64_t>(value));
  }

  /**
   * @brief Function to write a vector of unsigned integers to the internal
   * buffer
   */
  template <typename T>
  void field_packed_uint(int field, std::vector<T> const& value)
  {
    struct_size += p->put_uint(encode_field_number<std::vector<T>>(field));
    auto lpos = p->m_buff.size();
    p->put_byte(0);
    auto sz = std::accumulate(value.begin(), value.end(), 0, [p = this->p](size_t sum, auto val) {
      return sum + p->put_uint(val);
    });

    struct_size += sz + 1;
    for (; sz > 0x7f; sz >>= 7, struct_size++)
      p->m_buff.insert(p->m_buff.begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80));
    (p->m_buff)[lpos] = static_cast<uint8_t>(sz);
  }

  /**
   * @brief Function to write a blob to the internal buffer
   */
  template <typename T>
  void field_blob(int field, T const& values)
  {
    struct_size += p->put_uint(encode_field_number<T>(field));
    struct_size += p->put_uint(values.size());
    struct_size += p->put_bytes<typename T::value_type>(values);
  }

  /**
   * @brief Function to write a struct to the internal buffer
   */
  template <typename T>
  void field_struct(int field, T const& value)
  {
    struct_size += p->put_uint(encode_field_number(field, ProtofType::FIXEDLEN));
    auto lpos = p->m_buff.size();
    p->put_byte(0);
    auto sz = p->write(value);
    struct_size += sz + 1;
    for (; sz > 0x7f; sz >>= 7, struct_size++)
      p->m_buff.insert(p->m_buff.begin() + (lpos++), static_cast<uint8_t>((sz & 0x7f) | 0x80));
    (p->m_buff)[lpos] = static_cast<uint8_t>(sz);
  }

  /**
   * @brief Function to write a vector of strings to the internal buffer
   */
  void field_repeated_string(int field, std::vector<std::string> const& value)
  {
    for (auto const& elem : value)
      field_blob(field, elem);
  }

  /**
   * @brief Function to write a vector of structs to the internal buffer
   */
  template <typename T>
  void field_repeated_struct(int field, std::vector<T> const& value)
  {
    for (auto const& elem : value)
      field_struct(field, elem);
  }

  /**
   * @brief Function to write a vector of struct blobs to the internal
   * buffer
   */
  template <typename T>
  void field_repeated_struct_blob(int field, std::vector<T> const& value)
  {
    for (auto const& elem : value)
      field_blob(field, elem);
  }

  /**
   * @brief Returns the total length of the buffer written
   */
  size_t value() { return struct_size; }
};

}  // namespace cudf::io::orc::detail
