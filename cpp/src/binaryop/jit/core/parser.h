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
