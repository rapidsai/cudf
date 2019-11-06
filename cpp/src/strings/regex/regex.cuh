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

namespace cudf
{
class string_view;

namespace strings
{
namespace detail
{

struct Reljunk;
struct Reinst;

//
class dreclass
{
public:
    int32_t builtins{};
    int32_t count{};
    char32_t* chrs{};
    const uint8_t* codepoint_flags{};

    __device__ inline dreclass(const uint8_t* flags) : codepoint_flags(flags) {}
    __device__ bool is_match(char32_t ch);
};

//
class dreprog
{
    int32_t startinst_id, num_capturing_groups;
    int32_t insts_count, starts_count, classes_count;
    const uint8_t* codepoint_flags{};
    int32_t* startinst_ids{};
    Reinst* insts{};
    char32_t* classes{};
    void* relists_mem{};
    u_char* stack_mem1{};
    u_char* stack_mem2{};

    void free_relists();

    //
    __device__ inline int32_t regexec( string_view const& dstr, Reljunk& jnk, int32_t& begin, int32_t& end, int32_t groupid=0 );
    __device__ inline int32_t call_regexec( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t groupid=0 );

public:
    dreprog() = delete;
    ~dreprog() = default;
    dreprog(const dreprog&) = default;
    dreprog(dreprog&&) = default;
    dreprog& operator=(const dreprog&) = default;
    dreprog& operator=(dreprog&&) = default;

    //
    static dreprog* create_from(const char32_t* pattern, const uint8_t* cp_flags);
    static void destroy(dreprog*);

    //static std::unique_ptr<dreprog, std::function<void(dreprog*)>> create(const char32_t* pattern, const uint8_t* cp_flags);
    //void destroy();

    bool alloc_relists(size_t count);

    int32_t inst_counts();
    int32_t group_counts();

    __device__ inline void set_stack_mem(u_char* s1, u_char* s2);

    __host__ __device__ inline Reinst* get_inst(int32_t idx);
    __device__ inline int get_class(int32_t idx, dreclass& cls);
    __device__ inline int* get_startinst_ids();

    __device__ inline int find( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end );
    __device__ inline int extract( int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t col );

};

#define MAX_STACK_INSTS 1000

// 10128 ≈ 1000 instructions
// Formula is from data_size_for calculaton
// bytes = (8+2)*x + (x/8) = 10.125x < 11x  where x is number of instructions

#define RX_STACK_SMALL  112
#define RX_STACK_MEDIUM 1104
#define RX_STACK_LARGE  10128

#define RX_SMALL_INSTS  (RX_STACK_SMALL/11)
#define RX_MEDIUM_INSTS (RX_STACK_MEDIUM/11)
#define RX_LARGE_INSTS  (RX_STACK_LARGE/11)


} // namespace detail
} // namespace strings
} // namespace cudf

#include "./regex.inl"
