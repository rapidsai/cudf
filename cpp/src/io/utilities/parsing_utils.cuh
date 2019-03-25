#pragma once

#include <vector>

#include "cudf.h"

constexpr size_t max_chunk_bytes = 256*1024*1024; // 256MB

gdf_error countAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, 
	gdf_size_type &rec_cnt);

gdf_error findAllFromSet(const char *h_data, size_t h_size, std::vector<char> keys, cu_recstart_t result_offset,
	cu_recstart_t *recStart);