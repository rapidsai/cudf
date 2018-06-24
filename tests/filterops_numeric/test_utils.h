
#ifndef GDF_TEST_UTILS
#define GDF_TEST_UTILS

#include <string>
#include <functional>
#include <vector>
#include "gdf/gdf.h"

auto print_binary(unsigned char n) -> void ;

auto chartobin(unsigned char n) -> char *;

using CheckFunctionType =  void (char*, bool*, int);

auto check_column(gdf_column * column, CheckFunctionType check_function) -> void; 

auto print_column(gdf_column * column) -> void;

#endif // GDF_TEST_UTILS
