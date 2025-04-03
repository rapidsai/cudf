/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cstdint>
#include <string>

namespace cudf::io::parquet::detail {

// Max decimal precisions according to the parquet spec:
// https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#decimal
auto constexpr MAX_DECIMAL32_PRECISION  = 9;
auto constexpr MAX_DECIMAL64_PRECISION  = 18;
auto constexpr MAX_DECIMAL128_PRECISION = 38;  // log10(2^(sizeof(int128_t) * 8 - 1) - 1)

// Constants copied from arrow source and renamed to match the case
int32_t constexpr MESSAGE_DECODER_NEXT_REQUIRED_SIZE_INITIAL         = sizeof(int32_t);
int32_t constexpr MESSAGE_DECODER_NEXT_REQUIRED_SIZE_METADATA_LENGTH = sizeof(int32_t);
int32_t constexpr IPC_CONTINUATION_TOKEN                             = -1;
std::string const ARROW_SCHEMA_KEY                                   = "ARROW:schema";

// Schema type ipc message has zero length body
int64_t constexpr SCHEMA_HEADER_TYPE_IPC_MESSAGE_BODYLENGTH = 0;

// bit space we are reserving in column_buffer::user_data
constexpr uint32_t PARQUET_COLUMN_BUFFER_SCHEMA_MASK          = (0xff'ffffu);
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED = (1 << 24);
// if this column has a list parent anywhere above it in the hierarchy
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT = (1 << 25);

/**
 * @brief Count the number of leading zeros in an unsigned integer
 */
static inline int CountLeadingZeros32(uint32_t value)
{
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 32;
  return static_cast<int>(__builtin_clz(value));
#else
  int bitpos = 0;
  while (value != 0) {
    value >>= 1;
    ++bitpos;
  }
  return 32 - bitpos;
#endif
}

}  // namespace cudf::io::parquet::detail
