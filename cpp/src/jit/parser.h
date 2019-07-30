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
#include <set>
#include <map>
#include <vector>
#include <cudf/types.h>

/**
@brief Parse and Transform a piece of PTX code that contains the implementation 
of a `__device__` function into a CUDA `__device__` `__inline__` function.

@param `src` The input PTX code.

@param `function_name` The User defined function that the output CUDA function
will have.

@param `output_arg_type` The output type of the PTX function, e.g. "int", "int64_t"

@return The output CUDA `__device__` `__inline__` function
*/

namespace cudf {
namespace jit {

class ptx_parser {

private:
  
  std::string ptx;

  std::string function_name;

  std::string output_arg_type;

  std::set<int> pointer_arg_list;

  std::map<std::string, std::string> input_arg_list;

private:
  
  std::string parse_function_header(const std::string& src);
  
  std::string parse_param_list(const std::string& src);

  static std::string parse_param(const std::string& src);
  
  std::vector<std::string> parse_function_body(const std::string& src);
  
  std::string parse_statement(const std::string& src);
  
  std::string parse_instruction(const std::string& src);
  
  static std::string find_register_type(const std::string& src);
  
  static std::string register_type_to_cppname(const std::string& register_type);
  
  static std::string parse_register_type(const std::string& src);
  
  static std::string get_rid_of_nonalnum_sqrbra(const std::string& src);
  
  static std::string escape_percent(const std::string& src);

public:
  
  ptx_parser() = delete;

  ptx_parser(
    const std::string& ptx_,
    const std::string& function_name_,
    const std::string& output_arg_type_,
    const std::set<int>& pointer_arg_list_
  );

  // parse the source!!!
  std::string parse();

};


inline std::string parse_single_function_ptx(const std::string& src,
                                      const std::string& function_name,
                                      const std::string& output_arg_type,
                                      const std::set<int>& pointer_arg_list = {0}){
  
  ptx_parser instance(src, function_name, output_arg_type, pointer_arg_list);
  
  return instance.parse();

}

std::string parse_single_function_cuda(const std::string& src,
                                       const std::string& function_name,
                                       const std::string& output_arg_type);

}
}

#endif
