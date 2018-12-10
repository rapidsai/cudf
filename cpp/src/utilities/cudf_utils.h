#ifndef GDF_UTILS_H
#define GDF_UTILS_H

#include <cuda_runtime_api.h>
#include <vector>
#include "cudf.h"

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
inline 
gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size) { 
	return (( size + ( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE ); 
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Flatten AOS info from gdf_columns into SOA.
 * 
 * @Param[in] cols Host-side array of gdf_columns
 * @Param[in] ncols # columns
 * @Param[out] d_cols Pointer to device array of columns
 * @Param[out] d_types Device array of column types
 * 
 * @Returns
 */
/* ----------------------------------------------------------------------------*/
inline void soa_col_info(gdf_column* cols, size_t ncols, void** d_cols, int* d_types)
{
	std::vector<void*> v_cols(ncols, nullptr);
	std::vector<int>   v_types(ncols, 0);
	for(size_t i=0; i<ncols; ++i)
	{
		v_cols[i] = cols[i].data;
		v_types[i] = cols[i].dtype;
	}

	void** h_cols = &v_cols[0];
	int* h_types = &v_types[0];
	cudaMemcpy(d_cols, h_cols, ncols*sizeof(void*), cudaMemcpyHostToDevice);//TODO: add streams
	cudaMemcpy(d_types, h_types, ncols*sizeof(int), cudaMemcpyHostToDevice);//TODO: add streams
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Flatten AOS info from gdf_columns into SOA.
 * 
 * @Param[in] cols Host-side array of pointers to gdf_columns
 * @Param[in] ncols # columns
 * @Param[out] d_cols Pointer to device array of columns
 * @Param[out] d_valids Pointer to device array of gdf_valid_type for each column
 * @Param[out] d_types Device array of column types
 * 
 * @Returns
 */
/* ----------------------------------------------------------------------------*/
inline void soa_col_info(gdf_column** cols, size_t ncols, void** d_cols, gdf_valid_type** d_valids, int* d_types)
{
	std::vector<void*> v_cols(ncols, nullptr);
	std::vector<gdf_valid_type*> v_valids(ncols, nullptr);
	std::vector<int>  v_types(ncols, 0);
	for(size_t i=0; i<ncols; ++i)
	{
		v_cols[i] = cols[i]->data;
		v_valids[i] = cols[i]->valid;
		v_types[i] = cols[i]->dtype;
	}

	void** h_cols = &v_cols[0];
	gdf_valid_type** h_valids = &v_valids[0];
	int* h_types = &v_types[0];
	cudaMemcpy(d_cols, h_cols, ncols*sizeof(void*), cudaMemcpyHostToDevice);//TODO: add streams
	cudaMemcpy(d_valids, h_valids, ncols*sizeof(gdf_valid_type*), cudaMemcpyHostToDevice);//TODO: add streams
	cudaMemcpy(d_types, h_types, ncols*sizeof(int), cudaMemcpyHostToDevice);//TODO: add streams
}

#endif // GDF_UTILS_H