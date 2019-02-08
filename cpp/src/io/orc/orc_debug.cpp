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


#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "orc_debug.h"

#if defined(_WIN32) 
#include <windows.h>
#endif

namespace cudf {
namespace orc {

void defualtOutputMessageStringFunction(const char*  szBuffer);

FILE*           _output_stream                  = NULL;
kvFMessageFunc  _output_message_function        = defualtOutputMessageStringFunction;
unsigned int    _debugLevel = 0;

// ============================================================================================
void    kv_assert_entity(const char* _message, const char *_filename, unsigned int _line_number)
{
    kvOutputMessageFormat("Assertion Failed: [%s][L:%d], <%s>", _filename, _line_number, _message);

#if _WIN32
    __debugbreak();
#endif

    abort();
    return;
}    // kv_assert_entity

void defualtOutputMessageStringFunction(const char*  szBuffer)
{
    if( !_output_stream )_output_stream = stdout;
    fprintf(_output_stream, "%s\n", szBuffer);    
    fflush(_output_stream);
    
#if defined(_WIN32) && defined(_DEBUG) 
    OutputDebugStringA(szBuffer);
    OutputDebugStringA("\n");
#endif
};    // OutputMessageString


#if !defined(_MSC_VER) || _MSC_VER < 1400
#    define VSN_PRINTF(buffer_, sizeOfBuffer_, format_, argptr_)    vsnprintf(buffer_, sizeOfBuffer_, format_, argptr_)
#else    // for VC.net 2005
#    define VSN_PRINTF(buffer_, sizeOfBuffer_, format_, argptr_)    vsnprintf_s(buffer_, sizeOfBuffer_, _TRUNCATE, format_, argptr_)    
#endif

void kvOutputMessageFormat(const char*  messageFormat, ...)
{
#define cLargeBufferSize 512
    char szBuffer[cLargeBufferSize];
    va_list args;
    long nBuf;
    va_start(args, messageFormat);
    nBuf = VSN_PRINTF(szBuffer, cLargeBufferSize, messageFormat, args);
    va_end(args);

    if (nBuf==0){
        return;
    }

    if( nBuf > cLargeBufferSize )_output_message_function("kvOutputMessageFormat buffer over flow");

    _output_message_function(szBuffer);

} // defualtOutputMessage


kvFMessageFunc    kvSetMessageFunction(kvFMessageFunc func)
{
    kvFMessageFunc old_func = _output_message_function;
    _output_message_function = func;
    return old_func;
}    // kvSetMessageFunction


FILE*    kvSetMessageFile(FILE *fp)
{
    FILE* old_stream = _output_stream;
    _output_stream = fp;
    return old_stream;
}    // kvSetMessageFile


void    kvSetDebugLevel(unsigned int level)
{
    _debugLevel = level;    
};    // kvSetDebugLevel


unsigned int    kvGetDebugLevel()
{
    return _debugLevel;
}    // kvGetDebugLevel

}   // namespace orc
}   // namespace cudf