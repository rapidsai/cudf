#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include <gdf/gdf.h>

__device__
static
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
	if ( valid )
		return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
	else
		return true;
}


#endif
