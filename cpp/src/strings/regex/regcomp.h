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
#include <string>
#include <vector>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @brief Actions and Tokens (regex instruction types)
 * 
 * ```
 *	02xx are operators, value == precedence
 *	03xx are tokens, i.e. operands for operators
 * ```
 */
enum InstType
{
    CHAR    = 0177,  // Literal character
    RBRA    = 0201,  // Right bracket, )
    LBRA    = 0202,  // Left bracket, (
    OR      = 0204,  // Alternation, |
    ANY     = 0300,  // Any character except newline, .
    ANYNL   = 0301,  // Any character including newline, .
    BOL     = 0303,  // Beginning of line, ^
    EOL     = 0304,  // End of line, $
    CCLASS  = 0305,  // Character class, []
    NCCLASS = 0306,  // Negated character class, []
    BOW     = 0307,  // Boundary of word, /b
    NBOW    = 0310,  // Not boundary of word, /b
    END     = 0377   // Terminate: match found
};

/**
 * @brief Class type for regex compiler instruction.
 */
struct reclass
{
    int32_t builtins;    // bit mask identifying builtin classes
    std::u32string literals; // ranges as pairs of utf-8 characters
    reclass() : builtins(0) {}
    reclass(int m) : builtins(m) {}
};

/**
 * @brief Structure of an encoded regex instruction
 */
struct reinst
{
    int32_t	type;           /* operator type or instruction type */
    union	{
        int32_t	cls_id;     /* class pointer */
        char32_t c;         /* character */
        int32_t	subid;      /* sub-expression id for RBRA and LBRA */
        int32_t	right_id;   /* right child of OR */
    } u1;
    union {                 /* regexec relies on these two being in the same union */
        int32_t left_id;    /* left child of OR */
        int32_t next_id;    /* next instruction for CAT & LBRA */
    } u2;
    int32_t reserved4;
};

/**
 * @brief Regex program handles parsing a pattern in to individual set
 * of chained instructions.
 */
class reprog
{
 public:

    reprog() = default;
    reprog(const reprog&) = default;
    reprog(reprog&&) = default;
    ~reprog() = default;
    reprog& operator=(const reprog&) = default;
    reprog& operator=(reprog&&) = default;

    /**
     * @brief Parses the given regex pattern and compiles
     * into a list of chained instructions.
     */
    static reprog create_from(const char32_t* pattern);

    int32_t add_inst(int32_t type);
    int32_t add_inst(reinst inst);
    int32_t add_class(reclass cls);

    void set_groups_count(int32_t groups);
    int32_t groups_count() const;

    const reinst* insts_data() const;
    int32_t insts_count() const;
    reinst& inst_at(int32_t id);

    reclass& class_at(int32_t id);
    int32_t classes_count() const;

    const int32_t* starts_data() const;
    int32_t starts_count() const;

    void set_start_inst(int32_t id);
    int32_t get_start_inst() const;

    void optimize1();
    void optimize2();
    void print(); // for debugging

private:
    std::vector<reinst> _insts;
    std::vector<reclass> _classes;
    int32_t _startinst_id;
    std::vector<int32_t> _startinst_ids; // short-cut to speed-up ORs
    int32_t _num_capturing_groups;
};

} // namespace detail
} // namespace strings
} // namespace cudf
