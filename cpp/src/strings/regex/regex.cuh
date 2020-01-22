/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <memory>
#include <functional>

namespace cudf
{

class string_view;

namespace strings
{
namespace detail
{

struct reljunk;
struct reinst;
class reprog;

/**
 * @brief Regex class stored on the device and executed by reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
class reclass_device
{
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
class reprog_device
{
public:
    reprog_device() = delete;
    ~reprog_device() = default;
    reprog_device(const reprog_device&) = default;
    reprog_device(reprog_device&&) = default;
    reprog_device& operator=(const reprog_device&) = default;
    reprog_device& operator=(reprog_device&&) = default;

    /**
     * @brief Create device program instance from a regex pattern.
     *
     * The number of strings is needed to compute the state data size required when evaluating the regex.
     *
     * @param pattern The regex pattern to compile.
     * @param cp_flags The code-point lookup table for character types.
     * @param strings_count Number of strings that will be evaluated.
     * @param stream CUDA stream for asynchronous memory allocations.
     * @return The program device object.
     */
    static std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>
        create(std::string const& pattern, const uint8_t* cp_flags, int32_t strings_count, cudaStream_t stream=0);
    /**
     * @brief Called automatically by the unique_ptr returned from create().
     */
    void destroy();

    /**
     * @brief Returns the number of regex instructions.
     */
    int32_t insts_counts() const  { return _insts_count; }

    /**
     * @brief Returns the number of regex groups found in the expression.
     */
    int32_t group_counts() const  { return _num_capturing_groups; }

    /**
     * @brief This sets up the memory used for keeping track of the regex progress.
     *
     * Call this for each string before calling find or extract.
     */
    __device__ inline void set_stack_mem(u_char* s1, u_char* s2);

    /**
     * @brief Returns the regex instruction object for a given index.
     */
    __host__ __device__ inline reinst* get_inst(int32_t idx) const;

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
     * @param idx The string index used for mapping the state memory for this string in global memory (if necessary).
     * @param d_str The string to search.
     * @param[in,out] begin Position index to begin the search. If found, returns the position found in the string.
     * @param[in,out] end Position index to end the search. If found, returns the last position matching in the string.
     * @return Returns 0 if no match is found.
     */
    __device__ inline int32_t find( int32_t idx, string_view const& d_str, int32_t& begin, int32_t& end );

    /**
     * @brief Does an extract evaluation using the compiled expression on the given string.
     * 
     * This will find a specific match within the string when more than match occurs.
     *
     * @param idx The string index used for mapping the state memory for this string in global memory (if necessary).
     * @param d_str The string to search.
     * @param[in,out] begin Position index to begin the search. If found, returns the position found in the string.
     * @param[in,out] end Position index to end the search. If found, returns the last position matching in the string.
     * @param column The specific instance to return if more than one match is found.
     * @return Returns 0 if no match is found.
     */
    __device__ inline int32_t extract( int32_t idx, string_view const& d_str, int32_t& begin, int32_t& end, int32_t column );

private:
    int32_t _startinst_id, _num_capturing_groups;
    int32_t _insts_count, _starts_count, _classes_count;
    const uint8_t* _codepoint_flags{}; // table of character types
    reinst* _insts{};            // array of regex instructions
    int32_t* _startinst_ids{};   // array of start instruction ids
    reclass_device* _classes{};  // array of regex classes
    void* _relists_mem{};        // runtime relist memory for regexec
    u_char* _stack_mem1{};       // memory for relist object 1
    u_char* _stack_mem2{};       // memory for relist object 2

    /**
     * @brief Executes the regex pattern on the given string.
     */
    __device__ inline int32_t regexec( string_view const& d_str, reljunk& jnk, int32_t& begin, int32_t& end, int32_t groupid=0 );

    /**
     * @brief Utility wrapper to setup state memory structures for calling regexec
     */
    __device__ inline int32_t call_regexec( int32_t idx, string_view const& d_str, int32_t& begin, int32_t& end, int32_t groupid=0 );

    reprog_device(reprog&); // must use create()
};


// 10128 ≈ 1000 instructions
// Formula is based on relist::data_size_for() calculaton;
// Stack ≈ (8+2)*x + (x/8) = 10.125x < 11x  where x is number of instructions
constexpr int32_t MAX_STACK_INSTS = 1000;

constexpr int32_t RX_STACK_SMALL  = 112;
constexpr int32_t RX_STACK_MEDIUM = 1104;
constexpr int32_t RX_STACK_LARGE  = 10128;

constexpr int32_t RX_SMALL_INSTS  = (RX_STACK_SMALL/11);
constexpr int32_t RX_MEDIUM_INSTS = (RX_STACK_MEDIUM/11);
constexpr int32_t RX_LARGE_INSTS  = (RX_STACK_LARGE/11);

} // namespace detail
} // namespace strings
} // namespace cudf

#include "./regex.inl"
