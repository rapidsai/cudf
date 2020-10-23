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

namespace cudf {
namespace io {

__device__ void trim_zeros(char const *&raw_data, long &countNumber)
{
  char const *start_ptr = raw_data;
  while (*raw_data == '0') { raw_data++; }
  countNumber -= raw_data - start_ptr;
}

__device__ bool lex_compare_int64max(const char *raw_data)
{
  constexpr int length   = 19;
  const char int64_max[] = {
    '9', '2', '2', '3', '3', '7', '2', '0', '3', '6', '8', '5', '4', '7', '7', '5', '8', '0', '7'};
  for (int i = 0; i < length; ++i) {
    if (raw_data[i] < int64_max[i]) {
      return true;
    } else if (raw_data[i] > int64_max[i]) {
      return false;
    }
  }
  return true;
}

__device__ bool lex_compare_int64min(const char *raw_data)
{
  constexpr int length   = 19;
  const char int64_min[] = {
    '9', '2', '2', '3', '3', '7', '2', '0', '3', '6', '8', '5', '4', '7', '7', '5', '8', '0', '8'};
  for (int i = 0; i < length; ++i) {
    if (raw_data[i] < int64_min[i]) {
      return true;
    } else if (raw_data[i] > int64_min[i]) {
      return false;
    }
  }
  return true;
}

__device__ bool lex_compare_uint64max(const char *raw_data)
{
  constexpr int length    = 20;
  const char uint64_max[] = {'1', '8', '4', '4', '6', '7', '4', '4', '0', '7',
                             '3', '7', '0', '9', '5', '5', '1', '6', '1', '5'};
  // This function is only called on raw_data such that the number of digits in
  // raw_data is 20.
  for (int i = 0; i < length; ++i) {
    if (raw_data[i] < uint64_max[i]) {
      return true;
    } else if (raw_data[i] > uint64_max[i]) {
      return false;
    }
  }
  return true;
}

__device__ cudf::size_type *get_counter_address(char const *raw_data,
                                                long countNumber,
                                                column_info &stats)
{
  constexpr uint32_t Int64StringLength  = 19;
  constexpr uint32_t UInt64StringLength = 20;
  bool is_negative                      = (*raw_data == '-');
  // Skip parity sign
  raw_data += (is_negative || (*raw_data == '+'));

  if (countNumber < Int64StringLength) {  // CASE 0 : Accept validity
    // If the length of the string representing the integer is smaller
    // than string length of Int64Max then count this as an integer
    // representable by int64
    return is_negative ? &stats.negative_small_int_count : &stats.positive_small_int_count;
  } else {
    // Remove preceding zeros
    trim_zeros(raw_data, countNumber);
  }
  // After trimming the number of digits could be less than maximum
  // int64 digit count
  if (countNumber < Int64StringLength) {  // CASE 0 : Accept validity
    // If the length of the string representing the integer is smaller
    // than string length of Int64Max then count this as an integer
    // representable by int64
    return is_negative ? &stats.negative_small_int_count : &stats.positive_small_int_count;
  } else if (countNumber > UInt64StringLength) {  // CASE 1 : Reject validity
    // If the length of the string representing the integer is greater
    // than string length of UInt64Max then count this as a string
    // since it cannot be represented as an int64 or uint64
    return &stats.string_count;
  } else if (countNumber == UInt64StringLength && is_negative) {
    // A negative integer of length UInt64Max digit count cannot be represented
    // as a 64 bit integer
    return &stats.string_count;
  }

  if (countNumber == Int64StringLength && is_negative) {
    return lex_compare_int64max(raw_data) ? &stats.negative_small_int_count : &stats.string_count;
  } else if (countNumber == Int64StringLength && !is_negative) {
    return lex_compare_int64min(raw_data) ? &stats.positive_small_int_count : &stats.string_count;
  } else if (countNumber == UInt64StringLength) {
    return lex_compare_uint64max(raw_data) ? &stats.big_int_count : &stats.string_count;
  }

  return &stats.string_count;
}

}  // namespace io
}  // namespace cudf
