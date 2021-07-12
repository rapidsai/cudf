/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <thrust/optional.h>
#include <thrust/pair.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Return the integer component of a decimal string.
 *
 * This is reads everything up to the exponent 'e' notation.
 * The return includes the integer digits and any exponent offset.
 *
 * @param[in,out] iter Start of characters to parse
 * @param[in] end End of characters to parse
 * @return Integer component and exponent offset.
 */
__device__ inline thrust::pair<uint64_t, int32_t> parse_integer(char const*& iter,
                                                                char const* iter_end,
                                                                const char decimal_pt_char = '.')
{
  // highest value where another decimal digit cannot be appended without an overflow;
  // this preserves the most digits when scaling the final result
  constexpr uint64_t decimal_max = (std::numeric_limits<uint64_t>::max() - 9L) / 10L;

  uint64_t value     = 0;  // for checking overflow
  int32_t exp_offset = 0;
  bool decimal_found = false;

  while (iter < iter_end) {
    auto const ch = *iter++;
    if (ch == decimal_pt_char && !decimal_found) {
      decimal_found = true;
      continue;
    }
    if (ch < '0' || ch > '9') {
      --iter;
      break;
    }
    if (value > decimal_max) {
      exp_offset += static_cast<int32_t>(!decimal_found);
    } else {
      value = (value * 10) + static_cast<uint64_t>(ch - '0');
      exp_offset -= static_cast<int32_t>(decimal_found);
    }
  }
  return {value, exp_offset};
}

/**
 * @brief Return the exponent of a decimal string.
 *
 * This should only be called after the exponent 'e' notation was detected.
 * The return is the exponent (base-10) integer and can only be
 * invalid if `check_only == true` and invalid characters are found or the
 * exponent overflows an int32.
 *
 * @tparam check_only Set to true to verify the characters are valid and the
 *         exponent value in the decimal string does not overflow int32
 * @param[in,out] iter Start of characters to parse
 *                     (points to the character after the 'E' or 'e')
 * @param[in] end End of characters to parse
 * @return Integer value of the exponent
 */
template <bool check_only = false>
__device__ thrust::optional<int32_t> parse_exponent(char const* iter, char const* iter_end)
{
  constexpr uint32_t exponent_max = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());

  // get optional exponent sign
  int32_t const exp_sign = [&iter] {
    auto const ch = *iter;
    if (ch != '-' && ch != '+') { return 1; }
    ++iter;
    return (ch == '-' ? -1 : 1);
  }();

  // parse exponent integer
  int32_t exp_ten = 0;
  while (iter < iter_end) {
    auto const ch = *iter++;
    if (ch < '0' || ch > '9') {
      if (check_only) { return thrust::nullopt; }
      break;
    }

    uint32_t exp_check = static_cast<uint32_t>(exp_ten * 10) + static_cast<uint32_t>(ch - '0');
    if (check_only && (exp_check > exponent_max)) { return thrust::nullopt; }  // check overflow
    exp_ten = static_cast<int32_t>(exp_check);
  }

  return exp_ten * exp_sign;
}

/**
 * @brief Converts the string in the range [iter, iter_end) into a decimal.
 *
 * @tparam DecimalType The decimal type to be returned
 * @param iter The beginning of the string. Unless iter >= iter_end, iter is dereferenced
 * @param iter_end The end of the characters to parse
 * @param scale The scale to be applied
 * @return
 */
template <typename DecimalType>
__device__ DecimalType parse_decimal(char const* iter, char const* iter_end, int32_t scale)
{
  auto const sign = [&] {
    if (iter_end <= iter) { return 0; }
    if (*iter == '-') { return -1; }
    if (*iter == '+') { return 1; }
    return 0;
  }();

  // if string begins with a sign, continue with next character
  if (sign != 0) ++iter;

  auto [value, exp_offset] = parse_integer(iter, iter_end);
  if (value == 0) { return DecimalType{0}; }

  // check for exponent
  int32_t exp_ten = 0;
  if ((iter < iter_end) && (*iter == 'e' || *iter == 'E')) {
    ++iter;
    if (iter < iter_end) { exp_ten = parse_exponent<false>(iter, iter_end).value(); }
  }
  exp_ten += exp_offset;

  // shift the output value based on the exp_ten and the scale values
  if (exp_ten < scale) {
    value = value / static_cast<uint64_t>(exp10(static_cast<double>(scale - exp_ten)));
  } else {
    value = value * static_cast<uint64_t>(exp10(static_cast<double>(exp_ten - scale)));
  }

  return static_cast<DecimalType>(value) * (sign == 0 ? 1 : sign);
}
}  // namespace detail
}  // namespace strings
}  // namespace cudf
