# CMake script to generate C++ source file containing LTO-IR data
#
# This script is called by the build system to embed pre-compiled LTO-IR
# data into the libcudf binary.

# Convert semicolon-separated lists back to CMake lists
string(REPLACE ";" "|" LTO_IR_FILES_TEMP "${LTO_IR_FILES}")
string(REPLACE "|" ";" LTO_IR_FILES "${LTO_IR_FILES_TEMP}")

string(REPLACE ";" "|" OPERATOR_NAMES_TEMP "${OPERATOR_NAMES}")
string(REPLACE "|" ";" OPERATOR_NAMES "${OPERATOR_NAMES_TEMP}")

# Function to convert binary file to C++ byte array
function(file_to_hex_array FILE_PATH VAR_NAME OUTPUT_VAR)
  file(READ ${FILE_PATH} FILE_CONTENT HEX)
  
  # Split into byte-sized chunks
  string(REGEX MATCHALL ".." HEX_BYTES ${FILE_CONTENT})
  
  set(BYTE_ARRAY "")
  set(BYTE_COUNT 0)
  
  foreach(BYTE IN LISTS HEX_BYTES)
    if(BYTE_COUNT GREATER 0)
      string(APPEND BYTE_ARRAY ", ")
    endif()
    
    if(BYTE_COUNT GREATER 0 AND BYTE_COUNT EQUAL 16)
      string(APPEND BYTE_ARRAY "\n  ")
      set(BYTE_COUNT 0)
    endif()
    
    string(APPEND BYTE_ARRAY "0x${BYTE}")
    math(EXPR BYTE_COUNT "${BYTE_COUNT} + 1")
  endforeach()
  
  set(${OUTPUT_VAR} "{\n  ${BYTE_ARRAY}\n}" PARENT_SCOPE)
endfunction()

# Generate the C++ source file
set(GENERATED_CODE "/*
 * Auto-generated file containing pre-compiled LTO-IR data.
 * Do not modify manually.
 */

#include \"jit/lto_ir.hpp\"

#ifdef CUDF_USE_LTO_IR

#include <vector>
#include <string>

namespace cudf {
namespace jit {
namespace detail {

")

# Process each LTO-IR file
list(LENGTH LTO_IR_FILES FILE_COUNT)
math(EXPR FILE_COUNT_MINUS_1 "${FILE_COUNT} - 1")

foreach(INDEX RANGE ${FILE_COUNT_MINUS_1})
  list(GET LTO_IR_FILES ${INDEX} LTO_IR_FILE)
  list(GET OPERATOR_NAMES ${INDEX} OPERATOR_NAME)
  
  # Check if file exists
  if(NOT EXISTS "${LTO_IR_FILE}")
    message(WARNING "LTO-IR file not found: ${LTO_IR_FILE}")
    continue()
  endif()
  
  # Create variable name from operator name
  string(REPLACE "::" "_" VAR_NAME "${OPERATOR_NAME}")
  string(REPLACE " " "_" VAR_NAME "${VAR_NAME}")
  
  # Convert binary file to hex array
  file_to_hex_array("${LTO_IR_FILE}" "${VAR_NAME}" HEX_ARRAY)
  
  string(APPEND GENERATED_CODE "
// LTO-IR data for operator: ${OPERATOR_NAME}
static const unsigned char ${VAR_NAME}_data[] = ${HEX_ARRAY};

")
endforeach()

string(APPEND GENERATED_CODE "
// Registration function for built-in LTO-IR operators
void register_builtin_lto_ir_operators() {
  auto& cache = lto_ir_cache::instance();
  
")

# Generate registration calls
foreach(INDEX RANGE ${FILE_COUNT_MINUS_1})
  list(GET OPERATOR_NAMES ${INDEX} OPERATOR_NAME)
  
  # Create variable name from operator name
  string(REPLACE "::" "_" VAR_NAME "${OPERATOR_NAME}")
  string(REPLACE " " "_" VAR_NAME "${VAR_NAME}")
  
  string(APPEND GENERATED_CODE "  
  cache.register_operator(
    \"${OPERATOR_NAME}\",
    {std::string(reinterpret_cast<const char*>(${VAR_NAME}_data), sizeof(${VAR_NAME}_data))},
    {} // No dependencies for built-in operators
  );
")
endforeach()

string(APPEND GENERATED_CODE "
}

}  // namespace detail

void initialize_builtin_lto_ir_operators() {
  detail::register_builtin_lto_ir_operators();
}

}  // namespace jit
}  // namespace cudf

#endif // CUDF_USE_LTO_IR
")

# Write the generated code to the output file
file(WRITE "${OUTPUT_FILE}" "${GENERATED_CODE}")

message(STATUS "Generated LTO-IR data file: ${OUTPUT_FILE}")
message(STATUS "Included ${FILE_COUNT} LTO-IR operators")