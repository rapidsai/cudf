#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <core/utilities/error_utils.hpp>
#include <cudf/functions.h>

#include <cuda_runtime.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>

#include "rmm/rmm.h"

//TODO: I pulled this rom Andrei's sql ops code, we should consider putting this somewhere
//others can access this api
namespace{ //annonymus

  //helper function:
  //flatten AOS info from gdf_columns into SOA (2 arrays):
  //(1) column array pointers and (2) types;
  //
  gdf_error soa_col_info(gdf_column** cols, size_t ncols, void** d_cols, int* d_types, cudaStream_t stream)
  {
    std::vector<void*> v_cols(ncols,nullptr);
    std::vector<int>   v_types(ncols, 0);
    for(int i=0;i<ncols;++i)
      {
	v_cols[i] = cols[i]->data;
	v_types[i] = cols[i]->dtype;
      }

    void** h_cols = &v_cols[0];
    int* h_types = &v_types[0];
    CUDA_TRY( cudaMemcpyAsync(d_cols, h_cols, 
                              ncols*sizeof(void*), 
                              cudaMemcpyHostToDevice,stream) ); //TODO: add streams
    CUDA_TRY( cudaMemcpy(d_types, h_types, 
                         ncols*sizeof(int), 
                         cudaMemcpyHostToDevice,stream) ); //TODO: add streams
  }
}



gdf_error gpu_window_function(gdf_column ** window_order_columns, int num_window_order_columns, window_function_type window_type,
		gdf_column ** window_reduction_columns, int num_window_reduction_columns, window_reduction_type * reductions,
		gdf_column ** window_partition_columns, int num_window_partition_columns,
		order_by_type * order_by_types, char * workspace,gdf_size_type & workspace_size){

	//will always have at least one reduction
	gdf_size_type num_rows = window_reduction_columns[0].size;

	if(num_window_partition_columns > 0){

		//can be used to get the amount of space that should be preallocated for this
		if(workspace == nullptr){
			workspace_size = 0;
			if(num_window_partition_columns > 1){
				//more than one partition column so we will hash the data together
				workspace_size += sizeof(unsigned long long) * window_partition_columns[0].size + ((window_partition_columns[0].size + (GDF_VALID_BITSIZE -1 )) /GDF_VALID_BITSIZE));
			}
			return GDF_SUCCESS;
		}
	}

	//a gdf column

	cudaStream_t stream;
	cudaStreamCreate(&stream);


	//hash the partition columns
	gdf_column hashed_data;

	if(num_window_partition_columns > 0){
		gdf_hash_columns(window_partition_columns, num_window_partition_columns,
				&hashed_data, &stream);
		gdf_column-view(&data, (void *) data,
				(gdf_valid_type *) ( ((char *) data ) + (sizeof(unsigned long long) * num_rows))
				,window_partition_columns[0].size,GDF_INT64);

	}else{
		hashed_data.size = 0;
	}



	//stable sort backwards starting with the least significant order by
	/*template<typename IndexT>
__host__ __device__
void multi_col_order_by(size_t nrows,
			size_t ncols,
			void* const* d_cols,
			int* const  d_gdf_t,
			IndexT*      d_indx,
			cudaStream_t stream = NULL)*/

	void ** device_order_columns;
	RMM_TRY( RMM_ALLOC(&device_order_columns,sizeof(void *) * num_window_order_columns + 1, stream) );

	//copy copy device pointers

	int * device_column_types;
	RMM_TRY( RMM_ALLOC((void**)&device_column_types,sizeof(int) * num_window_order_columns + 1, stream) );

	gdf_column** order_by_cols = new gdf_column*[num_window_order_columns + 1];
	for(int i = 0; i < num_window_order_columns; i++){
		order_by_cols[i] = window_order_columns[i];
	}
	order_by_cols[num_window_order_columns] = &hashed_data;

	soa_col_info(order_by_cols, num_window_order_columns + 1,
				 device_order_columns, device_column_types, stream);

	gdf_size_type * device_index_outputs;
	RMM_TRY( RMM_ALLOC((void**)&device_index_outputs, sizeof(gdf_size_type) * num_rows, stream) );

	multi_col_order_by(num_rows, num_window_order_columns + 1, device_order_columns,
			           device_column_types, device_index_outputs, stream);

	//now we have our ordered arangement for the table
	//process reduction

	delete[] order_by_cols;
	RMM_TRY( RMM_FREE(device_order_columns, stream) );
	RMM_TRY( RMM_FREE(device_column_types, stream) );

	//no stable sort by the hash of the partition column

	//perform windowed functions here

	CUDA_TRY( cudaStreamSynchronize(stream) );
	CUDA_TRY( cudaStreamDestroy(stream) );
}


//so we have no segmented sorting in thrust, which means that one way to accomplish our goals is to stable sort in
//backwards order
//this will line up all of our columns nicely


//because our partitions are a group of columns hashing the values together makes for fewer operations later on when determining
//partitions

//so primitives we need are
//the sorting
//hashing
