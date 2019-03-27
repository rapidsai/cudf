#pragma once

#include <vector>

#include "cudf.h"

using ll_uint_t = unsigned long long int;

gdf_error countAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, 
	gdf_size_type &count);

template<class T>
gdf_error findAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, ll_uint_t result_offset,
	T *positions);