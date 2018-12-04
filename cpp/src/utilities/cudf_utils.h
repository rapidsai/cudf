#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include "cudf.h"
#include "miscellany.hpp"

#ifdef __CUDACC__
__host__ __device__
#endif
inline
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
	if ( valid )
		return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
	else
		return true;
}

#ifdef __CUDACC__
__host__ __device__
#endif
/**
 * Calculates the size in bytes of a validity indicator pseudo-column for a given column's size.
 *
 * @note Actually, this is the size in bytes of a column of bits, where the individual
 * bit-container elements are of the same size as `gdf_valid_type`.
 *
 * @param[in] column_size the number of elements, i.e. the number of bits to be available
 * for use, in the column
 * @return the number of bytes necessary to make available for the validity indicator pseudo-column
 */
inline
gdf_size_type get_number_of_bytes_for_valid(gdf_size_type column_size) {
    return gdf::util::div_rounding_up_safe(column_size, GDF_VALID_BITSIZE);
}

#endif
