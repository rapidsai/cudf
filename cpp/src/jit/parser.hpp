/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace CUDF_EXPORT cudf {

namespace jit {

struct ptx_param {
  std::string register_type;
  std::string identifier;
};

/**
 * @brief Parse and transform a piece of PTX code that contains the implementation
 * of a `__device__` function into a CUDA `__device__` `__inline__` function.
 *
 * @param `src` The input PTX code.
 * @param `function_name` The User defined function that the output CUDA function
 * will have.
 * @param `param_types` The types of the parameters for the the kernel. This will be used to specify
 * the CUDA function declaration for the wrapped PTX code.
 * @return The output CUDA `__device__` `__inline__` function
 */
class ptx_parser {
 private:
  std::string ptx;

  std::string function_name;

  std::map<unsigned int, std::string> param_types;

 private:
  static ptx_param parse_param(std::string_view param_decl);

  /**
   * @brief parse the parameter list of the PTX code
   *
   * @param src The input parameter list part of the PTX code
   * @return The parsed CUDA parameter list
   */
  static std::vector<ptx_param> parse_params(std::string_view src);

  /**
   * @brief parse the PTX function header
   *
   * @param src The header part of the PTX code
   * @return The parsed CUDA header

   As an example:

    .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eff(
      .param .b64 _ZN8__main__7add$241Eff_param_0,
      .param .b32 _ZN8__main__7add$241Eff_param_1,
      .param .b32 _ZN8__main__7add$241Eff_param_2
    )

   will be parsed into:

        {{"b64", "_ZN8__main__7add_241Eff_param_0"},
         {"b32", "_ZN8__main__7add_241Eff_param_1"},
         {"b32", "_ZN8__main__7add_241Eff_param_2"}}

   */
  static std::vector<ptx_param> parse_function_header(std::string_view src);

  /**
   * @brief Create a CUDA function header to wrap a parsed PTX.
   *
   *
   * The result always has `__device__ __inline__ void`. The types of the
   * parameters are determined from, in descending order of priority:
   *  1. The types provided in the `param_types` map
   *  2. The register types of the parameters found in the PTX function header. i.e. register types
   * `s64` and `b64` would map to `long int`, and `s32` and `b32` would map to `int`
   *
   * @param function_name the function name to use in the CUDA code
   * @param ptx_params the parsed info of each param found in the PTX
   * @param param_types the CUDA source code types to use in place of the ptx parameter register
   * type
   */
  static std::string to_cuda_function_header(
    std::string_view function_name,
    std::vector<ptx_param> const& ptx_params,
    std::map<unsigned int, std::string> const& param_types);

  /**
   * @brief Convert the input PTX instruction into an inline PTX
   * statement without changing (exceptions exist).
   *
   * Non-alphanumeric characters in register identifiers, except underscores, are replaced with
   underscore. Example:
   *
   *  fma.rn.f32 	%f4, %f3, %f1, %f2
   *
   *    ---> asm volatile ("  fma.rn.f32 _f4, _f3, _f1, _f2;");
   *
   * See the document at https://github.com/hummingtree/cudf/wiki/PTX-parser
   * for the detailed description about the exceptions.
   *
   * @param src The statement to be parsed.
   * @return The resulting CUDA inline PTX statement.
   */
  static std::string parse_instruction(std::string_view src);

  /**
   * @brief Remove leading white characters and call `parse_instruction`.
   *
   * @param src The statement to be parsed.
   * @return The resulting CUDA statement.
   */
  static std::string parse_statement(std::string_view src);

  /**
   * @brief parse function body of the PTX code into statements by `;`s.
   *
   * @param src The function body of the PTX code
   * @return The parsed statements
   */
  static std::vector<std::string> parse_function_body(std::string_view src);

 public:
  ptx_parser() = delete;

  /**
   * @brief Constructor of the `ptx_parser` class
   *
   * @param ptx_ The input PTX code that contains the function whose
   * CUDA is to be generated.
   * @param function_name_ The function name of the output CUDA function
   *
   * @param param_types the types of the parameters of the function. each entry should map the
   * parameter index to the CUDA type of the parameter
   */
  ptx_parser(std::string ptx_,
             std::string function_name_,
             std::map<unsigned int, std::string> param_types);

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
 * @param param_types the types of the parameters of the function. each entry should map the
 * parameter index to the CUDA type of the parameter
 * @return The output CUDA device function
 */
inline std::string parse_single_function_ptx(std::string_view src,
                                             std::string_view function_name,
                                             std::map<unsigned int, std::string> const& param_types)
{
  ptx_parser instance(std::string{src}, std::string{function_name}, param_types);

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
std::string parse_single_function_cuda(std::string_view src, std::string_view function_name);

}  // namespace jit
}  // namespace CUDF_EXPORT cudf
