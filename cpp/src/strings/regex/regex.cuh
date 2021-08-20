/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <strings/regex/regcomp.h>

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>

#include <functional>
#include <memory>

namespace cudf {

class string_view;

namespace strings {
namespace detail {

struct reljunk;
struct reinst;
class reprog;

using match_pair   = thrust::pair<cudf::size_type, cudf::size_type>;
using match_result = thrust::optional<match_pair>;

constexpr int32_t RX_STACK_SMALL  = 112;    ///< fastest stack size
constexpr int32_t RX_STACK_MEDIUM = 1104;   ///< faster stack size
constexpr int32_t RX_STACK_LARGE  = 10128;  ///< fast stack size
constexpr int32_t RX_STACK_ANY    = 8;      ///< slowest: uses global memory

/**
 * @brief Mapping the number of instructions to device code stack memory size.
 *
 * ```
 * 10128 ≈ 1000 instructions
 * Formula is based on relist::data_size_for() calculation;
 * Stack ≈ (8+2)*x + (x/8) = 10.125x < 11x  where x is number of instructions
 * ```
 */
constexpr int32_t RX_SMALL_INSTS  = (RX_STACK_SMALL / 11);
constexpr int32_t RX_MEDIUM_INSTS = (RX_STACK_MEDIUM / 11);
constexpr int32_t RX_LARGE_INSTS  = (RX_STACK_LARGE / 11);

/**
 * @brief Regex class stored on the device and executed by reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
class reclass_device {
 public:
  int32_t builtins{};
  int32_t count{};
  char32_t* literals{};

  __device__ bool is_match(char32_t ch, const uint8_t* flags);
};

/**
 * @brief Regex program of instructions/data for a specific regex pattern.
 *
 * Once create, this find/extract methods are used to evaluating the regex instructions
 * against a single string.
 */
class reprog_device {
 public:
  reprog_device()                     = delete;
  ~reprog_device()                    = default;
  reprog_device(const reprog_device&) = default;
  reprog_device(reprog_device&&)      = default;
  reprog_device& operator=(const reprog_device&) = default;
  reprog_device& operator=(reprog_device&&) = default;

  /**
   * @brief Create device program instance from a regex pattern.
   *
   * The number of strings is needed to compute the state data size required when evaluating the
   * regex.
   *
   * @param pattern The regex pattern to compile.
   * @param cp_flags The code-point lookup table for character types.
   * @param strings_count Number of strings that will be evaluated.
   * @param stream CUDA stream for asynchronous memory allocations. To ensure correct
   * synchronization on destruction, the same stream should be used for all operations with the
   * created objects.
   * @return The program device object.
   */
  static std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> create(
    std::string const& pattern,
    const uint8_t* cp_flags,
    int32_t strings_count,
    rmm::cuda_stream_view stream);

  /**
   * @brief Called automatically by the unique_ptr returned from create().
   */
  void destroy();

  /**
   * @brief Returns the number of regex instructions.
   */
  __host__ __device__ int32_t insts_counts() const { return _insts_count; }

  /**
   * @brief Returns true if this is an empty program.
   */
  __device__ bool is_empty() const { return insts_counts() == 0 || get_inst(0)->type == END; }

  /**
   * @brief Returns the number of regex groups found in the expression.
   */
  __host__ __device__ inline int32_t group_counts() const { return _num_capturing_groups; }

  /**
   * @brief Returns the regex instruction object for a given index.
   */
  __device__ inline reinst* get_inst(int32_t idx) const;

  /**
   * @brief Returns the regex class object for a given index.
   */
  __device__ inline reclass_device get_class(int32_t idx) const;

  /**
   * @brief Returns the start-instruction-ids vector.
   */
  __device__ inline int32_t* startinst_ids() const;

  /**
   * @brief Does a find evaluation using the compiled expression on the given string.
   *
   * @tparam stack_size One of the `RX_STACK_` values based on the `insts_count`.
   * @param idx The string index used for mapping the state memory for this string in global memory
   * (if necessary).
   * @param d_str The string to search.
   * @param[in,out] begin Position index to begin the search. If found, returns the position found
   * in the string.
   * @param[in,out] end Position index to end the search. If found, returns the last position
   * matching in the string.
   * @return Returns 0 if no match is found.
   */
  template <int stack_size>
  __device__ inline int32_t find(int32_t idx,
                                 string_view const& d_str,
                                 int32_t& begin,
                                 int32_t& end);

  /**
   * @brief Does an extract evaluation using the compiled expression on the given string.
   *
   * This will find a specific match within the string when more than match occurs.
   * The find() function should be called first to locate the begin/end bounds of the
   * the matched section.
   *
   * @tparam stack_size One of the `RX_STACK_` values based on the `insts_count`.
   * @param idx The string index used for mapping the state memory for this string in global memory
   * (if necessary).
   * @param d_str The string to search.
   * @param begin Position index to begin the search. If found, returns the position found
   * in the string.
   * @param end Position index to end the search. If found, returns the last position
   * matching in the string.
   * @param group_id The specific group to return its matching position values.
   * @return If valid, returns the character position of the matched group in the given string,
   */
  template <int stack_size>
  __device__ inline match_result extract(cudf::size_type idx,
                                         string_view const& d_str,
                                         cudf::size_type begin,
                                         cudf::size_type end,
                                         cudf::size_type group_id);

 private:
  int32_t _startinst_id, _num_capturing_groups;
  int32_t _insts_count, _starts_count, _classes_count;
  const uint8_t* _codepoint_flags{};  // table of character types
  reinst* _insts{};                   // array of regex instructions
  int32_t* _startinst_ids{};          // array of start instruction ids
  reclass_device* _classes{};         // array of regex classes
  void* _relists_mem{};               // runtime relist memory for regexec

  /**
   * @brief Executes the regex pattern on the given string.
   */
  __device__ inline int32_t regexec(
    string_view const& d_str, reljunk& jnk, int32_t& begin, int32_t& end, int32_t group_id = 0);

  /**
   * @brief Utility wrapper to setup state memory structures for calling regexec
   */
  template <int stack_size>
  __device__ inline int32_t call_regexec(
    int32_t idx, string_view const& d_str, int32_t& begin, int32_t& end, int32_t group_id = 0);

  reprog_device(reprog&);  // must use create()
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf

#include "./regex.inl"
