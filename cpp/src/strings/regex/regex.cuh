/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cudf/strings/regex/flags.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>

#include <memory>

namespace cudf {

class string_view;

namespace strings {
namespace detail {

struct relist;

using match_pair   = thrust::pair<cudf::size_type, cudf::size_type>;
using match_result = thrust::optional<match_pair>;

constexpr int32_t MAX_SHARED_MEM      = 2048;  ///< Memory size for storing prog instruction data
constexpr std::size_t MAX_WORKING_MEM = 0x01FFFFFFFF;  ///< Memory size for state data
constexpr int32_t MINIMUM_THREADS     = 256;  // Minimum threads for computing working memory

/**
 * @brief Regex class stored on the device and executed by reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
struct alignas(16) reclass_device {
  int32_t builtins{};
  int32_t count{};
  char32_t const* literals{};

  __device__ inline bool is_match(char32_t const ch, uint8_t const* flags) const;
};

/**
 * @brief Regex program of instructions/data for a specific regex pattern.
 *
 * Once created, the find/extract methods are used to evaluate the regex instructions
 * against a single string.
 *
 * An instance of the class requires working memory for evaluating the regex
 * instructions for the string. Determine the size of the required memory by
 * calling either `working_memory_size()` or `compute_strided_working_memory()`.
 * Once the buffer is allocated, pass the device pointer to the `set_working_memory()`
 * member function.
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
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return The program device object.
   */
  static std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> create(
    std::string const& pattern, rmm::cuda_stream_view stream);

  /**
   * @brief Create the device program instance from a regex pattern.
   *
   * @param pattern The regex pattern to compile.
   * @param re_flags Regex flags for interpreting special characters in the pattern.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return The program device object.
   */
  static std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> create(
    std::string const& pattern, regex_flags const re_flags, rmm::cuda_stream_view stream);

  /**
   * @brief Called automatically by the unique_ptr returned from create().
   */
  void destroy();

  /**
   * @brief Returns the number of regex instructions.
   */
  [[nodiscard]] CUDF_HOST_DEVICE int32_t insts_counts() const { return _insts_count; }

  /**
   * @brief Returns the number of regex groups found in the expression.
   */
  [[nodiscard]] CUDF_HOST_DEVICE inline int32_t group_counts() const
  {
    return _num_capturing_groups;
  }

  /**
   * @brief Returns true if this is an empty program.
   */
  [[nodiscard]] __device__ inline bool is_empty() const;

  /**
   * @brief Returns the size needed for working memory for the given thread count.
   *
   * @param num_threads Number of threads to be executed in parallel
   * @return Size of working memory in bytes
   */
  [[nodiscard]] std::size_t working_memory_size(int32_t num_threads) const;

  /**
   * @brief Compute working memory for the given thread count with a maximum size.
   *
   * The `min_rows` overrules the `requested_max_size`.
   * That is, the `requested_max_size` may be
   * exceeded to keep the number of rows greater than `min_rows`.
   * Also, if `rows < min_rows` then `min_rows` is not enforced.
   *
   * @param rows Number of rows to execute in parallel
   * @param min_rows The least number of rows to meet `max_size`
   * @param requested_max_size Requested maximum bytes for the working memory
   * @return The size of the working memory and the number of parallel rows it will support
   */
  [[nodiscard]] std::pair<std::size_t, int32_t> compute_strided_working_memory(
    int32_t rows,
    int32_t min_rows               = MINIMUM_THREADS,
    std::size_t requested_max_size = MAX_WORKING_MEM) const;

  /**
   * @brief Set the device working memory buffer to use for the regex execution.
   *
   * @param buffer Device memory pointer.
   * @param thread_count Number of threads the memory buffer will support.
   * @param max_insts Set to the maximum instruction count if reusing the
   *                  memory buffer for other regex calls.
   */
  void set_working_memory(void* buffer, int32_t thread_count, int32_t max_insts = 0);

  /**
   * @brief Returns the size of shared memory required to hold this instance.
   *
   * This can be called on the CPU for specifying the shared-memory size in the
   * kernel launch parameters.
   * This may return 0 if the MAX_SHARED_MEM value is exceeded.
   */
  [[nodiscard]] int32_t compute_shared_memory_size() const;

  /**
   * @brief Returns the thread count passed on `set_working_memory`.
   */
  [[nodiscard]] __device__ inline int32_t thread_count() const { return _thread_count; }

  /**
   * @brief Store this object into the given device pointer (e.g. shared memory).
   *
   * No data is stored if MAX_SHARED_MEM is exceeded for this object.
   */
  __device__ inline void store(void* buffer) const;

  /**
   * @brief Load an instance of this class from a device buffer (e.g. shared memory).
   *
   * Data is loaded from the given buffer if MAX_SHARED_MEM is not exceeded for the given object.
   * Otherwise, a copy of the object is returned.
   */
  [[nodiscard]] __device__ static inline reprog_device load(reprog_device const prog, void* buffer);

  /**
   * @brief Does a find evaluation using the compiled expression on the given string.
   *
   * @param thread_idx The index used for mapping the state memory for this string in global memory.
   * @param d_str The string to search.
   * @param[in,out] begin Position index to begin the search. If found, returns the position found
   * in the string.
   * @param[in,out] end Position index to end the search. If found, returns the last position
   * matching in the string.
   * @return Returns 0 if no match is found.
   */
  __device__ inline int32_t find(int32_t const thread_idx,
                                 string_view const d_str,
                                 cudf::size_type& begin,
                                 cudf::size_type& end) const;

  /**
   * @brief Does an extract evaluation using the compiled expression on the given string.
   *
   * This will find a specific match within the string when more than match occurs.
   * The find() function should be called first to locate the begin/end bounds of the
   * the matched section.
   *
   * @param thread_idx The index used for mapping the state memory for this string in global memory.
   * @param d_str The string to search.
   * @param begin Position index to begin the search. If found, returns the position found
   * in the string.
   * @param end Position index to end the search. If found, returns the last position
   * matching in the string.
   * @param group_id The specific group to return its matching position values.
   * @return If valid, returns the character position of the matched group in the given string,
   */
  __device__ inline match_result extract(int32_t const thread_idx,
                                         string_view const d_str,
                                         cudf::size_type begin,
                                         cudf::size_type end,
                                         cudf::size_type const group_id) const;

 private:
  struct reljunk {
    relist* __restrict__ list1;
    relist* __restrict__ list2;
    int32_t starttype{};
    char32_t startchar{};

    __device__ inline reljunk(relist* list1, relist* list2, reinst const inst);
    __device__ inline void swaplist();
  };

  /**
   * @brief Returns the regex instruction object for a given id.
   */
  __device__ inline reinst get_inst(int32_t id) const;

  /**
   * @brief Returns the regex class object for a given id.
   */
  __device__ inline reclass_device get_class(int32_t id) const;

  /**
   * @brief Executes the regex pattern on the given string.
   */
  __device__ inline int32_t regexec(string_view const d_str,
                                    reljunk jnk,
                                    cudf::size_type& begin,
                                    cudf::size_type& end,
                                    cudf::size_type const group_id = 0) const;

  /**
   * @brief Utility wrapper to setup state memory structures for calling regexec
   */
  __device__ inline int32_t call_regexec(int32_t const thread_idx,
                                         string_view const d_str,
                                         cudf::size_type& begin,
                                         cudf::size_type& end,
                                         cudf::size_type const group_id = 0) const;

  reprog_device(reprog&);

  int32_t _startinst_id;          // first instruction id
  int32_t _num_capturing_groups;  // instruction groups
  int32_t _insts_count;           // number of instructions
  int32_t _starts_count;          // number of start-insts ids
  int32_t _classes_count;         // number of classes
  int32_t _max_insts;             // for partitioning working memory

  uint8_t const* _codepoint_flags{};  // table of character types
  reinst const* _insts{};             // array of regex instructions
  int32_t const* _startinst_ids{};    // array of start instruction ids
  reclass_device const* _classes{};   // array of regex classes

  std::size_t _prog_size{};  // total size of this instance
  void* _buffer{};           // working memory buffer
  int32_t _thread_count{};   // threads available in working memory
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf

#include "./regex.inl"
