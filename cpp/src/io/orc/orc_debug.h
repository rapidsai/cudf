/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef Included__DEBUG_COMMON_H__
#define Included__DEBUG_COMMON_H__

#ifdef    _DEBUG
#define DEBUG_EVALUATION
#endif

#include <stdio.h>
#include <errno.h>

namespace cudf {
namespace orc {

extern void             kv_assert_entity(const char* _message, const char *_filename, unsigned int _line_number);

extern void             kvOutputMessageFormat(const char*  messageFormat, ...);

extern void             kvSetDebugLevel(unsigned int level);
extern unsigned int     kvGetDebugLevel();

extern FILE*            kvSetMessageFile(FILE* fp);

typedef void (*kvFMessageFunc)(const char* message);

extern kvFMessageFunc    kvSetMessageFunction(kvFMessageFunc func);

#define PRINTF    kvOutputMessageFormat

#define E_SYS_CHECK_(msg)                                        \
    {                                                            \
        int err_no = errno;                                      \
        PRINTF("System function error detected[0x%x]: %s",       \
            err_no, kvGetErrorNoString_(err_no));                \
        D_EXIT(msg);                                             \
    }    // E_SYS_CHECK_


#define NOOP                                        ((void)0)    //<    No operand macro
#define EXIT(message)                               kv_assert_entity(message, __FILE__, __LINE__);

#ifdef    DEBUG_EVALUATION
#    define D_MSG(...)                              kvOutputMessageFormat(__VA_ARGS__)
#    define D_MSG_N(n, ...)                         if(n < kvGetDebugLevel()) D_MSG(__VA_ARGS__)
#    define D_EXIT(message)                         kv_assert_entity(message, __FILE__, __LINE__);
#    define D_EXIT_MSG(...)                         D_MSG(__VA_ARGS__); D_EXIT("D_EXIT_MSG")
#    define D_ASSERT(_expression)                   if(!(_expression)){D_EXIT(#_expression);}
#    define D_ASSERT_MSG(_expression, ...)          if(!(_expression)){D_EXIT_MSG(__VA_ARGS__);}
#    define D_EXEC(_expression)                     _expression;
#    define E_CHECK(_expression)                    D_ASSERT(_expression)
#    define E_SYS_CHECK(_expression)                if((_expression)<0){E_SYS_CHECK_(#_expression);};
#    define E_CHECK_MSG(_expression, ...)           D_ASSERT_MSG(_expression, __VA_ARGS__)
#    define E_SUCCESS(_expression)                  D_ASSERT(kvError_Success == _expression)
#    define E_CHECK_ERRNO(_expression)              _expression; if( errno )D_EXIT_MSG("errno =%d: assertion:<%s>", errno, #_expression)
#else
#    define D_MSG(...)                              NOOP;
#    define D_MSG_N(n, ...)                         NOOP;
#    define D_EXIT(message)                         NOOP;
#    define D_EXIT_MSG(...)                         NOOP;
#    define D_ASSERT(_expression)                   NOOP;
#    define D_ASSERT_MSG(_expression, ...)          NOOP;
#    define D_EXEC(_expression)                     NOOP;
#    define E_SYS_CHECK(_expression)                _expression;
#    define E_CHECK(_expression)                    _expression;
#    define E_CHECK_MSG(_expression, ...)           _expression;
#    define E_SUCCESS(_expression)                  _expression;
#    define E_CHECK_ERRNO(_expression)              _expression;
#endif    

#define    D_MSG0(...)    D_MSG_N(0, __VA_ARGS__)
#define    D_MSG1(...)    D_MSG_N(1, __VA_ARGS__)
#define    D_MSG2(...)    D_MSG_N(2, __VA_ARGS__)
#define    D_MSG3(...)    D_MSG_N(3, __VA_ARGS__)

#define D_ENUM_RANGE_CHECK(x, enum_name)                D_ASSERT(x < enum_name##_InternalUse || "enum RANGE error");
#define D_RANGE_CHECK(x, lower_limit, upper_limit)      D_ASSERT(lower_limit <= x  ||  x  <= upper_limit || "value RANGE error");

}   // namespace orc
}   // namespace cudf

#endif    // Included__DEBUG_COMMON_H__

