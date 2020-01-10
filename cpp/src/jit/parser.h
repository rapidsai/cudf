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

#ifndef GDF_JIT_PARSER_H
#define GDF_JIT_PARSER_H

#include <string>
#include <set>
#include <map>
#include <vector>

namespace cudf {
namespace jit {

/**
 * @brief Parse and transform a piece of PTX code that contains the implementation 
 * of a `__device__` function into a CUDA `__device__` `__inline__` function.
 * 
 * @param `src` The input PTX code.
 * @param `function_name` The User defined function that the output CUDA function
 * will have.
 * @param `output_arg_type` The output type of the PTX function, e.g. "int", "int64_t"
 * @return The output CUDA `__device__` `__inline__` function
*/
class ptx_parser {

private:
  
  std::string ptx;

  std::string function_name;

  std::string output_arg_type;

  std::set<int> pointer_arg_list;

  std::map<std::string, std::string> input_arg_list;

private:
 
  /**
   * @brief parse and transform header part of the PTX code into a CUDA header
   *
   * The result always has `__device__ __inline__ void`. The types of the input
   * parameters are determined from, in descending order of priority:
   *  1. The first parameter is always of type "`output_arg_type`*"
   *  2. All other parameters marked in pointer_arg_list are of type "const void*"
   *  3. For parameters that are used in the function body their types are 
   *      inferred from their corresponding parameter loading instructions
   *  4. Unused parameters are always of type "int"
   *
   * @param src The header part of the PTX code
   * @return The parsed CUDA header
   
   As an example:

    .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eff(
	    .param .b64 _ZN8__main__7add$241Eff_param_0,
	    .param .b32 _ZN8__main__7add$241Eff_param_1,
	    .param .b32 _ZN8__main__7add$241Eff_param_2
    )

   will be transformed to

    __device__ __inline__ void GENERIC_BINARY_OP(
        float* _ZN8__main__7add_241Eff_param_0, 
        float _ZN8__main__7add_241Eff_param_1, 
        float _ZN8__main__7add_241Eff_param_2
    )
   
   */ 
  std::string parse_function_header(const std::string& src);
  
  /**
   * @brief parse and transform input parameter list of the PTX code into the 
   * corresponding CUDA form
   * 
   * @param src The input parameter list part of the PTX code
   * @return The parsed CUDA input parameter list
   */ 
  std::string parse_param_list(const std::string& src);
  
  /**
   * @brief parse and transform an input parameter line of the PTX code into the
   * corresponding CUDA form
   * 
   * @param src The input parameter line of the PTX code
   * @return The parsed CUDA input parameter
   */ 
  static std::string parse_param(const std::string& src);

  /**
   * @brief parse function body of the the PTX code into statements by `;`s.
   * 
   * @param src The function body of the PTX code
   * @return The parsed statements
   */ 
  std::vector<std::string> parse_function_body(const std::string& src);
  
  /**
   * @brief Remove leading white chractors and call `parse_instruction`.
   * 
   * @param src The statement to be parsed.
   * @return The resulting CUDA statement.
   */ 
  std::string parse_statement(const std::string& src);

  /**
   * @brief Convert the input PTX instruction into a inline PTX
   * statement without changing (exceptions exist).
   * 
   * Non-alphanemric that are not underscore in register identifiers 
   * are all replaced with underscore. An example:
   * 
   *  fma.rn.f32 	%f4, %f3, %f1, %f2
   *
   *    ---> asm volatile ("  fma.rn.f32 _f4, _f3, _f1, _f2;");
   *
   * If a regiter from the input parameters list is used in an instruction
   * its type is inferred from the intruction and saved in the `input_arg_list`
   * to be used in when parsing the function header.
   * 
   * See the document at https://github.com/hummingtree/cudf/wiki/PTX-parser
   * for the detailed description about the exceptions.
   *
   * @param src The statement to be parsed.
   * @return The resulting CUDA inline PTX statement.
   */ 
  std::string parse_instruction(const std::string& src);
  
  /**
   * @brief Convert register type (e.g. ".f32") to the corresponding
   * C++ type (e.g. "float")
   *
   * See the implementation for details
   * 
   * @param src The input code
   * @return The resulting code
   */ 
  static std::string register_type_to_cpp_type(const std::string& register_type);
  
  /**
   * @brief Convert register type (e.g. ".f32") to the corresponding
   * constraint in inline PTX syntax (e.g. "f")
   *
   * See the implementation for details
   * 
   * @param src The input code
   * @return The resulting code
   */ 
  static std::string register_type_to_contraint(const std::string& src);
   
  /**
   * @brief Replace any non-alphanumeric charactors that are not underscore with 
   * underscore. The leading `[` and trailing `]` are exempted, e.g.
   * 
   *  "[t$5]" --> "[t_5]"
   * 
   * @param src The input code
   * @return The resulting code
   */ 
  static std::string remove_nonalphanumeric(const std::string& src);
   
  /**
   * @brief Replace leading `%` in register identifiers with `_`.
   * 
   * According to PTX document `%` can only appear at the start of a register
   * identifier. At the same time `%` is not allowed in inline PTX. This function
   * first looks for the register identifier and if it starts with `%` replaces it
   * with `_`.
   * 
   * @param src The input code
   * @return The resulting code
   */ 
  static std::string escape_percent(const std::string& src);

public:
  
  ptx_parser() = delete;

  /**
   * @brief C'tor of the `ptx_parser` class
   * 
   * @param ptx_ The input PTX code that contains the function whose
   * CUDA is to be generated.
   * @param function_name_ The function name of the output CUDA function
   * @param output_arg_type_ The C++ type of the output parameter of the
   * function.
   * @param pointer_arg_list_ A list of the parameters that are pointers.
   * @return The resulting code
   */ 
  ptx_parser(
    const std::string& ptx_,
    const std::string& function_name_,
    const std::string& output_arg_type_,
    const std::set<int>& pointer_arg_list_
  );

  // parse the source!!!
  std::string parse();

};

 /**
  * @brief Parse and Transform a piece of PTX code that contains the implementation 
  * of a device function into a CUDA device function.
  * 
  * @param src The input PTX code.
  * @param function_name The User defined function that the output CUDA function
  * will have.
  * @param output_arg_type output_arg_type The C++ type of the output parameter of the
  * function
  * @param pointer_arg_list A list of the parameters that are pointers.
  * @return The output CUDA device function
  */
inline std::string parse_single_function_ptx(const std::string& src,
                                      const std::string& function_name,
                                      const std::string& output_arg_type,
                                      const std::set<int>& pointer_arg_list = {0}){
  
  ptx_parser instance(src, function_name, output_arg_type, pointer_arg_list);
  
  return instance.parse();

}

 /**
  * @brief In a piece of CUDA code that contains the implementation 
  * of a device function, locate the function and replace its function name
  * with the specified one.
  * 
  * @param src The input CUDA code.
  * @param function_name The User defined function that the output CUDA function
  * will have.
  * @return The output CUDA device function
  */
std::string parse_single_function_cuda(const std::string& src,
                                       const std::string& function_name);

}
}

#endif
