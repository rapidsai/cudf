#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include <gdf/gdf.h>

__host__ __device__
static
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
	if ( valid )
		return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
	else
		return true;
}

inline gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size) { return (( size + ( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE ); }

#endif
