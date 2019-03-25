#pragma once

#include <vector>

#include "cudf.h"

constexpr size_t max_chunk_bytes = 256*1024*1024; // 256MB

using ll_uint_t = unsigned long long int;

gdf_error countAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, 
	gdf_size_type &rec_cnt);

gdf_error findAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, ll_uint_t result_offset,
	ll_uint_t *recStart);