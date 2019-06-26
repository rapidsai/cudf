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

constexpr char percent_escape[] = "_";

inline bool is_white(const char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

inline std::map<std::string, std::string>& get_input_arg_list() {
  static std::map<std::string, std::string> m;
  return m;
}

std::string escape_percent(const std::string& src) {
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

std::string get_rid_of_nonalnum_sqrbra(const std::string& src) {
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

std::string parse_register_type(const std::string& src) {
  if (src == ".u16" || src == ".s16" || src == ".b16" || src == ".f16")
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
    return "x";
}

std::string register_type_to_cppname(const std::string& register_type) {
  if (register_type == ".u16")
    return "short";
  else if (register_type == ".s16")
    return "short";
  else if (register_type == ".f16")
    return "half";
  else if (register_type == ".u32")
    return "int";
  else if (register_type == ".s32")
    return "int";
  else if (register_type == ".f16x2")
    return "half2";
  else if (register_type == ".u64")
    return "long";
  else if (register_type == ".s64")
    return "long";
  else if (register_type == ".f32")
    return "float";
  else if (register_type == ".f64")
    return "double";
  else
    return "x";
}

std::string find_register_type(const std::string& src) {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;
  while (start < length) {
    while (start < length && src[start] != '.') {
      start++;
    }
    stop = start + 1;
    while (stop < length && src[stop] != '.') {
      stop++;
    }

    std::string code =
        register_type_to_cppname(std::string(src, start, stop - start));
    if (code != "x") {
      return code;
    } else {
      start = stop;
    }
  }
  printf("Cannot determine point type!\n");
  exit(1);
}

std::string parse_instruction(const std::string& src) {
  // I am assumming for an instruction statement the starting phrase is an
  // instruction.
  const size_t length = src.size();
  std::string output;
  std::string suffix;

  std::string original_code = "\n  /** Input ptx: \n    " + src + "\n  */\n";

  size_t start = 0;
  size_t stop = 0;
  bool is_instruction = true;
  std::string constraint;
  std::string register_type;
  bool blank = true;
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
             src[start] != '{' && src[start] != '}') {
        stop++;
      }
    }
    std::string piece = std::string(src, start, stop - start);
    if (is_instruction) {
      if (piece.find("ld.param") != std::string::npos) {
        register_type = std::string(piece, 8, stop - 8);
        // This is the ld.param sentence
        if (register_type_to_cppname(register_type) == "int") {
          // The trick to support statement like "ld.param.s32 %rd0, [...];",
          // where %rd0 is a 64-bit register.
          output += " cvt.s32.s32";
        } else {
          output += " mov" + register_type;
        }
        constraint = parse_register_type(register_type);
      } else if (piece.find("st.param") != std::string::npos) {
        return "// Out port does not support return value!" +
               original_code;  // Our port does not support return value;
      } else {
        output += " " + piece;
      }
      is_instruction = false;
    } else {
      // Here it should be the registers.
      if (piece.find("_param_") != std::string::npos) {
        output += " %0";
        suffix = ": : \"" + constraint + "\"(" +
                 get_rid_of_nonalnum_sqrbra(piece) + ")";
        // Here we get to see the actual type of the input arguments.
        get_input_arg_list()[get_rid_of_nonalnum_sqrbra(piece)] =
            register_type_to_cppname(register_type);
      } else {
        output += escape_percent(std::string(src, start, stop - start));
      }
    }
    start = stop;
  }
  if (!blank) output += ";";
  return "asm volatile (\"" + output + "\"" + suffix + ");" + original_code;
}

std::string parse_statement(const std::string& src) {
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

std::vector<std::string> parse_function_body(const std::string& src,
                                             const char indicator = ';') {
  const size_t length = src.size();
  size_t start = 0;
  size_t stop = 0;

  std::vector<std::string> statements;

  while (stop < length) {
    stop = start;
    while (stop < length && src[stop] != indicator) {
      stop++;
    }
    // statements.push_back(std::string(src, start, stop-start));
    statements.push_back(
        parse_statement(std::string(src, start, stop - start)));
    // parse_sentence();
    stop++;
    start = stop;
  }
  return statements;
}

std::string parse_param(const std::string& src, bool first = false) {
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
      name = get_rid_of_nonalnum_sqrbra(std::string(src, start, stop - start));
    }
    start = stop;
  }
  return name;
}

std::string parse_param_list(const std::string& src) {
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
    item_count++;
    if (item_count == 1) {  // The first input argument is always a pointer.
      first_name = parse_param(std::string(src, start, stop - start));
    } else {
      std::string name = parse_param(std::string(src, start, stop - start));
      arg_type = get_input_arg_list()[name];
      output += ", \n  " + arg_type + " " + name;
    }
    stop++;
    start = stop;
  }

  return "\n  " + arg_type + "* " + first_name + output + "\n";
}

std::string parse_function_header(const std::string& src,
                                  const std::string& function_name) {
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
  // std::string function_name = get_rid_of_nonalnum_sqrbra( std::string(src,
  // start, stop-start) );

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
std::string parse_single_function_ptx(const std::string& src,
                                      const std::string& function_name) {
  std::string no_comments = remove_comments(src);

  get_input_arg_list().clear();
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
      parse_function_header(function_header, function_name);

  std::string final_output = function_header_output + "\n";
  for (int i = 0; i < function_body_output.size(); i++) {
    if (function_body_output[i].find("ret;") != std::string::npos) {
      continue;
    }
    final_output += "  " + function_body_output[i] + "\n";
  }

  final_output += "}";

  return final_output;
}
