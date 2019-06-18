/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef GDF_BINARY_OPERATION_JIT_CORE_PARSER_H
#define GDF_BINARY_OPERATION_JIT_CORE_PARSER_H

#include <string>

/**
@brief Parse and Transform a piece of PTX code that contains the implementation 
of a `__device__` function into a CUDA `__device__` `__inline__` function.

@param `src` The input PTX code.

@param `function_name` The User defined function that the output CUDA function
will have.

@return The output CUDA `__device__` `__inline__` function
*/

std::string parse_single_function_ptx(const std::string& src,
                                      const std::string& function_name);

#endif
