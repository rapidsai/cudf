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

#include <cctype>
#include <map>
#include <string>
#include <vector>
#include <utilities/error_utils.hpp>

#include "parser.h"

namespace cudf {
namespace jit {

constexpr char percent_escape[] = "_";

inline bool is_white(const char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

std::string ptx_parser::escape_percent(const std::string& src) {
  // Since we are transforming into inline ptx we are not allowed to have
  // register names starting with %.
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;
  while (start < length && (is_white(src[start]) || src[start] == '[')) {
    start++;
  }
  stop = start;
  while (stop < length && !is_white(src[stop]) && src[stop] != ']') {
    stop++;
  }
  if (src[start] == '%') {
    std::string output = src;
    output.replace(start, 1, percent_escape);
    return output;
  } else {
    return src;
  }
}

std::string ptx_parser::remove_nonalphanumeric(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;
  std::string output = src;
  while (start < length && (is_white(src[start]) || src[start] == '[')) {
    start++;
  }
  stop = start;
  while (stop < length && !is_white(src[stop]) && src[stop] != ']') {
    if (!isalnum(src[stop]) && src[stop] != '_') {
      output[stop] = '_';
    }
    stop++;
  }
  return output.substr(start, stop - start);
}

std::string ptx_parser::register_type_to_contraint(const std::string& src) {
  if (src == ".b8" || src == ".u8" || src == ".s8")
    return "h";
  else if (src == ".u16" || src == ".s16" || src == ".b16" || src == ".f16")
    return "h";
  else if (src == ".b32" || src == ".u32" || src == ".s32" || src == ".f16x2")
    return "r";
  else if (src == ".u64" || src == ".b64" || src == ".s64")
    return "l";
  else if (src == ".f32")
    return "f";
  else if (src == ".f64")
    return "d";
  else
    return "x_reg";
}

std::string ptx_parser::register_type_to_cpp_type(const std::string& register_type) {
  if (register_type == ".b8" || register_type == ".s8" || register_type == ".u8")
    return "char";
  else if (register_type == ".u16")
    return "short int";
  else if (register_type == ".s16")
    return "short int";
  else if (register_type == ".f16")
    return "half";
  else if (register_type == ".u32")
    return "int";
  else if (register_type == ".s32")
    return "int";
  else if (register_type == ".f16x2")
    return "half2";
  else if (register_type == ".u64")
    return "long int";
  else if (register_type == ".s64")
    return "long int";
  else if (register_type == ".f32")
    return "float";
  else if (register_type == ".f64")
    return "double";
  else
    return "x_cpptype";
}

std::string ptx_parser::parse_instruction(const std::string& src) {
  // I am assumming for an instruction statement the starting phrase is an
  // instruction.
  const size_t length = src.size();
  std::string output;
  std::string suffix;

  std::string original_code = "\n   /**   " + src + "  */\n";

  int piece_count = 0;

  size_t start = 0;
  size_t stop = 0;
  bool is_instruction = true;
  bool is_param_loading_instruction = false;
  std::string constraint;
  std::string register_type;
  bool blank = true;
  std::string cpp_typename;
  while (stop < length) {
    while (start < length &&
           (is_white(src[start]) || src[start] == ',' || src[start] == '{' ||
            src[start] == '}')) {  // running to the first non-white character.
      if (src[start] == ',') output += ',';
      if (src[start] == '{') output += '{';
      if (src[start] == '}') output += '}';
      start++;
    }
    stop = start;
    if (stop < length) {
      blank = false;
      output += " ";
    } else {
      break;
    }
    if (src[start] == '[') {
      while (stop < length && src[stop] != ']') {
        stop++;
      }
      stop++;
    } else {
      while (stop < length && !is_white(src[stop]) && src[stop] != ',' &&
             src[stop] != ':') {
        stop++;
      }
      if(src[stop] == ':'){
        // This is a branch
        stop++;
        output += std::string(src, start, stop - start);
        start = stop;
        continue;
      }
    }
    std::string piece = std::string(src, start, stop - start);
    if (is_instruction) {
      if (piece.find("ld.param") != std::string::npos) {
        is_param_loading_instruction = true;
        register_type = std::string(piece, 8, stop - 8);
        // This is the ld.param sentence
        cpp_typename = register_type_to_cpp_type(register_type);
        if (cpp_typename == "int" || cpp_typename == "short int" || cpp_typename == "char") {
          // The trick to support `ld` statement whose destination reg. wider than 
          // the instruction width, e.g.
          //      
          //  "ld.param.s32 %rd0, [...];",
          //
          // where %rd0 is a 64-bit register. Directly converting to "mov" instruction
          // does not work since "register widening" is ONLY allowed for 
          // "ld", "st", and "cvt". So we use cvt instead and something like
          // "cvt.s32.s32". This keep the same operation behavior and when compiling to
          // SASS code "usually" (in cases I have seen) this is optimized away, thus
          // gives no performance panelty.
          output += " cvt" + register_type + register_type;
        } else {
          output += " mov" + register_type;
        }
        constraint = register_type_to_contraint(register_type);
      } else if (piece.find("st.param") != std::string::npos) {
        return "// Out port does not support return value!" +
               original_code;  // Our port does not support return value;
      } else if (piece[0] == '@') {
        output += " @" + remove_nonalphanumeric(piece.substr(1, piece.size()-1));
      } else {
        output += " " + piece;
      }
      is_instruction = false;
    } else {
      // Here it should be the registers.
      if (piece_count == 2 && is_param_loading_instruction) {
        // This is the source of the parameter loading instruction
        output += " %0";
        if(cpp_typename == "char"){
          suffix = ": : \"" + constraint + "\"( static_cast<short>(" +
                 remove_nonalphanumeric(piece) + "))";
        }else{  
          suffix = ": : \"" + constraint + "\"(" +
                 remove_nonalphanumeric(piece) + ")";
        }
        // Here we get to see the actual type of the input arguments.
        input_arg_list[remove_nonalphanumeric(piece)] =
            register_type_to_cpp_type(register_type);
      } else {
        output += escape_percent(std::string(src, start, stop - start));
      }
    }
    start = stop;
    piece_count++;
  }
  if (!blank) output += ";";
  return "asm volatile (\"" + output + "\"" + suffix + ");" + original_code;
}

std::string ptx_parser::parse_statement(const std::string& src) {
  // First find the first non-white charactor.
  const size_t length = src.size();
  size_t start = 0;
  while (start < length && is_white(src[start])) {
    start++;
  }

  if (start == length) {
    // got nothing
    return " \n";
  } else {
    // instruction
    return parse_instruction(std::string(src, start, length - start));
  }
}

std::vector<std::string> ptx_parser::parse_function_body(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;

  std::vector<std::string> statements;

  while (stop < length) {
    stop = start;
    while (stop < length && src[stop] != ';') {
      stop++;
    }
    statements.push_back(
        parse_statement(std::string(src, start, stop - start)));
    stop++;
    start = stop;
  }
  return statements;
}

std::string ptx_parser::parse_param(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;

  std::string name;

  int item_count = 0;
  while (stop < length) {
    while (start < length && is_white(src[start])) {
      start++;
    }
    stop = start;
    while (stop < length && !is_white(src[stop])) {
      stop++;
    }
    item_count++;
    if (item_count == 3) {
      name = remove_nonalphanumeric(std::string(src, start, stop - start));
    }
    start = stop;
  }
  return name;
}

std::string ptx_parser::parse_param_list(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;

  std::string output;

  int item_count = 0;
  std::string first_name;
  std::string arg_type;
  while (stop < length) {
    while (stop < length && src[stop] != ',') {
      stop++;
    }
    std::string name = parse_param(std::string(src, start, stop - start));
    if(pointer_arg_list.find(item_count) != pointer_arg_list.end()){
      if(item_count == 0){
        output += output_arg_type + "* " + name;
      }else{
        // On a 64-bit machine inside the PTX function body a pointer is
        // literally just a uint_64 so here is doesn't make sense to
        // have the type of the pointer. Thus we will just use void* here.
        output += ",\n  const void* " + name;
      }
    }else{
      if(input_arg_list.count(name)){
        output += ", \n  " + input_arg_list[name] + " " + name;
      }else{
        // This parameter isn't used in the function body so we just pretend
        // it's an int. After being inlined they are gone anyway.
        output += ", \n  int " + name;
      }
    }
    stop++;
    start = stop;
    item_count++;
  }

  return "\n  " + output + "\n";
}

std::string ptx_parser::parse_function_header(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;

  // Essentially we only need the information inside the two pairs of
  // parentices.
  while (start < length && is_white(src[start])) {
    start++;
  }

  if (src[start] == '(') {  // This function has a return type
    // First Pass: output param list
    while (start < length && src[start] != '(') {
      start++;
    }
    start++;
    stop = start;
    while (stop < length && src[stop] != ')') {
      stop++;
    }
    stop++;
  }

  start = stop;
  // The function name
  while (start < length && is_white(src[start])) {
    start++;
  }
  stop = start;
  while (stop < length && !is_white(src[stop]) && src[stop] != '(') {
    stop++;
  }

  start = stop;
  // Second Pass: input param list
  while (start < length && src[start] != '(') {
    start++;
  }
  start++;
  stop = start;
  while (stop < length && src[stop] != ')') {
    stop++;
  }
  std::string input_arg =
      parse_param_list(std::string(src, start, stop - start));
  return "\n__device__ __inline__ void " + function_name + "(" + input_arg +
         "){" + "\n";
}

std::string remove_comments(const std::string& src) {
  // Remove the comments in the input ptx code.
  size_t start = 0;
  size_t stop = 0;
  std::string output = src;
  while (start < output.size() - 1) {
    if (output[start] == '/' && output[start + 1] == '*') {
      stop = start + 2;
      while (stop < output.size() - 1) {
        if (output[stop] == '*' && output[stop + 1] == '/') {
          stop += 2;
          break;
        } else {
          stop++;
        }
      }
      output.erase(start, stop - start);
    } else if (output[start] == '/' && output[start + 1] == '/') {
      stop = start + 2;
      while (stop < output.size()) {
        if (output[stop] == '\n') {
          // stop += 1; // Keep the newline here.
          break;
        } else {
          stop++;
        }
      }
      output.erase(start, stop - start);
    } else {
      start++;
    }
  }
  return output;
}

// The interface
std::string ptx_parser::parse() {

  std::string no_comments = remove_comments(ptx);

  input_arg_list.clear();
  // Go directly to the .func mark
  const size_t length = no_comments.size();
  size_t start = no_comments.find(".func") + 5;
  if (start == length + 5) {
    printf("No function (.func) found in the input ptx code.\n");
    exit(1);
  }
  size_t stop = start;
  while (stop < length && no_comments[stop] != '{') {
    stop++;
  }

  std::string function_header = std::string(no_comments, start, stop - start);

  stop++;
  start = stop;

  int bra_count = 0;
  while (stop < length) {
    if (no_comments[stop] == '{') bra_count++;
    if (no_comments[stop] == '}') {
      if (bra_count == 0) {
        break;
      } else {
        bra_count--;
      }
    }
    stop++;
  }

  std::vector<std::string> function_body_output =
      parse_function_body(std::string(no_comments, start, stop - start));

  std::string function_header_output =
      parse_function_header(function_header);

  std::string final_output = function_header_output + "\n";
  final_output += " asm volatile (\"{\");";
  for (int i = 0; i < function_body_output.size(); i++) {
    if (function_body_output[i].find("ret;") != std::string::npos) {
      final_output += "  asm volatile (\"bra RETTGT;\");\n";
      continue;
    }
    final_output += "  " + function_body_output[i] + "\n";
  }
  final_output += " asm volatile (\"RETTGT:}\");";

  final_output += "}";

  return final_output;
}

ptx_parser::ptx_parser(
    const std::string& ptx_,
    const std::string& function_name_,
    const std::string& output_arg_type_,
    const std::set<int>& pointer_arg_list_
): 
  ptx(ptx_), 
  function_name(function_name_), 
  output_arg_type(output_arg_type_),
  pointer_arg_list(pointer_arg_list_) {}

// The interface
std::string parse_single_function_cuda(const std::string& src,
                                       const std::string& function_name){
  std::string no_comments = remove_comments(src);

  // For CUDA device function we just need to find the function 
  // name and replace it with the specified one.
  const size_t length = no_comments.size();
  size_t start = 0;
  size_t stop = start;
  
  while (stop < length && no_comments[stop] != '(') {
    stop++;
  }
  CUDF_EXPECTS(stop != length && stop != 0,
    "No CUDA device function found in the input CUDA code.\n");
  
  stop--;

  while (stop > 0 && is_white(no_comments[stop]) ){
    stop--;
  }
  CUDF_EXPECTS(stop != 0 || !is_white(no_comments[0]),
    "No CUDA device function name found in the input CUDA code.\n");
  
  start = stop;
  while (start > 0 && !is_white(no_comments[start])){
    start--;
  }
  start++;
  stop++;
  CUDF_EXPECTS(start < stop,
    "No CUDA device function name found in the input CUDA code.\n");
 
  no_comments.replace(start, stop-start, function_name);

  return no_comments;
}

} // namespace cudf
} // namespace jit
