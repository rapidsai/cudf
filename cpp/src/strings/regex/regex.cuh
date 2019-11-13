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

struct Reljunk;
struct Reinst;
class Reprog;

/**
 * @brief Regex class stored on the device and executed by Reprog_device.
 *
 * This class holds the unique data for any regex CCLASS instruction.
 */
class Reclass_device
{
public:
    int32_t builtins{};
    int32_t count{};
    char32_t* chrs{};

    __device__ bool is_match(char32_t ch, const uint8_t* flags);
};

/**
 * @brief Regex program of instructions/data for a specific regex pattern.
 *
 * This structure executes on the GPU.
 */
class Reprog_device
{
public:
    Reprog_device() = delete;
    ~Reprog_device() = default;
    Reprog_device(const Reprog_device&) = default;
    Reprog_device(Reprog_device&&) = default;
    Reprog_device& operator=(const Reprog_device&) = default;
    Reprog_device& operator=(Reprog_device&&) = default;

    // create instance from regex pattern
    static std::unique_ptr<Reprog_device, std::function<void(Reprog_device*)>>
     create(const char32_t* pattern, const uint8_t* cp_flags, int32_t strings_count, cudaStream_t stream=0);
    // called automatically by the unique_ptr returned from create()
    void destroy();

    int32_t insts_counts()   { return _insts_count; }
    int32_t group_counts()   { return _num_capturing_groups; }

    __device__ inline void set_stack_mem(u_char* s1, u_char* s2);

    __host__ __device__ inline Reinst* get_inst(int32_t idx);
    __device__ inline Reclass_device get_class(int32_t idx);
    __device__ inline int32_t* get_startinst_ids();

    __device__ inline int find( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end );
    __device__ inline int extract( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t col );

private:
    int32_t _startinst_id, _num_capturing_groups;
    int32_t _insts_count, _starts_count, _classes_count;
    const uint8_t* _codepoint_flags{}; // table of character types
    Reinst* _insts{};           // array of regex instructions
    int32_t* _startinst_ids{};   // array of start instruction ids
    Reclass_device* _classes{};  // array of regex classes
    void* _relists_mem{};        // runtime relist memory for regexec
    u_char* _stack_mem1{};       // memory for Relist object 1
    u_char* _stack_mem2{};       // memory for Relist object 2

    // executes the regex pattern on the given string
    __device__ inline int32_t regexec( string_view const& dstr, Reljunk& jnk, int32_t& begin, int32_t& end, int32_t groupid=0 );
    // utility wrapper to setup stack structures for calling regexec
    __device__ inline int32_t call_regexec( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t groupid=0 );

    Reprog_device(Reprog&);
};


// 10128 ≈ 1000 instructions
// Formula is from data_size_for calculaton
// bytes = (8+2)*x + (x/8) = 10.125x < 11x  where x is number of instructions
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
