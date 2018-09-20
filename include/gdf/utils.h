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

inline gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size) 
{ 
  return (( size + ( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE ); 
}

__host__ __device__
inline void turn_bit_on(gdf_valid_type * masks, gdf_index_type pos)
{
  if(nullptr != masks)
  {
    masks[pos/8] |= (gdf_valid_type(1) << (pos % 8));
  }
}


#endif
