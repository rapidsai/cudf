/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
	 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file gdf-csr.cu  code to convert a GDF matrix into a CSR
 *
 */

#include <cudf/cudf.h>
#include <cudf/utilities/error.hpp>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

using namespace std;

//--- all the private functions
template<typename T>
gdf_error runConverter(gdf_column **gdfData, csr_gdf *csrReturn, cudf::size_type * offsets);


//--- private CUDA functions / kernels
template<typename T>
__global__ void cudaCreateCSR(void *data, cudf::valid_type *valid, gdf_dtype dtype, int colID, T *A, int64_t *JA, cudf::size_type *offsets, cudf::size_type numRows);

__global__ void determineValidRecCount(cudf::valid_type *validArray, cudf::size_type numRows, cudf::size_type numCol, cudf::size_type * offset);

template<typename T>
__device__ T convertDataElement(gdf_column *gdf, int idx, gdf_dtype dtype);

__device__ int whichBitmapCSR(int record) { return (record/8);  }
__device__ int whichBitCSR(int bit) { return (bit % 8);  }
__device__ int checkBitCSR(cudf::valid_type data, int bit) {

	cudf::valid_type bitMask[8] 		= {1, 2, 4, 8, 16, 32, 64, 128};
	return (data & bitMask[bit]);
}


//
//------------------------------------------------------------
//

/*
 * Convert a Dense GDF into a CSR GDF
 *
 * Restrictions:  All columns need to be of the same length
 */
/**
 * @brief convert a GDF into a CSR
 *
 * Take a matrix in GDF format and convert it into a CSR.  The column major matrix needs to have every column defined.
 * Passing in a COO datset will be treated as a two column matrix
 *
 * @param[in] gdfData the ordered list of columns
 * @param[in] numCol the number of columns in the gdfData array
 *
 * @param[out] csrReturn a pointer to the returned data structure
 *
 * @return gdf_error code
 */
gdf_error gdf_to_csr(gdf_column **gdfData, int numCol, csr_gdf *csrReturn) {

	int64_t			numNull = 	0;
	int64_t			nnz		= 	0;
	cudf::size_type	numRows =	gdfData[0]->size;
	gdf_dtype		dType	=	gdf_dtype::GDF_invalid;		// the data type to make the CSR element array (A)

	/**
	 * Currently the gdf_dtype enum is arranged based on data size, as long as it stays that way the enum values can be
	 * exploited by just picking the largest enum value
	 *
	 * While looping, also get the number of null values (this will work one day)
	 */
	for ( int x =0; x < numCol; x++) {
		if( gdfData[x]->dtype > dType)
			dType = gdfData[x]->dtype;

		numNull += gdfData[x]->null_count;
	}

	if (dType == gdf_dtype::GDF_invalid || dType == gdf_dtype::GDF_STRING )
		return gdf_error::GDF_UNSUPPORTED_DTYPE;

	// the number of valid elements is simple the max number of possible elements (rows * columns) minus the number of nulls
	// the current problem is that algorithms are not setting null_count;
	// cudf::size_type is 32bits (int) but the total size could be larger than an int, so use a long
	nnz = (numRows * numCol) - numNull;

	// Allocate space for the offset - this will eventually be IA - dtype is long since the sum of all column elements could be larger than int32
	cudf::size_type * offsets;
    RMM_TRY(RMM_ALLOC((void**)&offsets, (numRows + 2) * sizeof(int64_t), 0)); // TODO: non-default stream?
    CUDA_TRY(cudaMemset(offsets, 0, ( sizeof(int64_t) * (numRows + 2) ) ));

    // do a pass over each columns, and have each column updates the row count
	//-- threads and blocks
    int threads = 1024;
    int blocks  = (numRows + threads - 1) / threads;
	for ( int x = 0; x < numCol; x++ ) {
		determineValidRecCount<<<blocks, threads>>>(gdfData[x]->valid, numRows, numCol, offsets);
	}


	//--------------------------------------------------------------------------------------
	// Now do an exclusive scan to compute the offsets for where to write data
    thrust::exclusive_scan(rmm::exec_policy()->on(0), offsets, (offsets + numRows + 1), offsets);

	//--------------------------------------------------------------------------------------
    // get the number of elements - NNZ, this is the last item in the array
    CUDA_TRY( cudaMemcpy((void *)&nnz, (void *)&offsets[numRows], sizeof(int64_t), cudaMemcpyDeviceToHost) );

	if ( nnz == 0)
		return GDF_CUDA_ERROR;

	//--------------------------------------------------------------------------------------
	// now start creating output data
    cudf::size_type* IA;
    RMM_TRY(RMM_ALLOC((void**)&IA, (numRows + 2) * sizeof(cudf::size_type), 0));
    CUDA_TRY(cudaMemcpy(IA, offsets, ( sizeof(cudf::size_type) * (numRows + 2) ), cudaMemcpyDeviceToDevice) );

    int64_t * 	JA;
    RMM_TRY( RMM_ALLOC((void**)&JA, (sizeof(int64_t) * nnz), 0));

    //----------------------------------------------------------------------------------
    // Now just missing A and the moving of data

	csrReturn->dtype	= dType;
	csrReturn->rows		= numRows;
	csrReturn->cols		= numCol;
	csrReturn->dtype	= dType;
    csrReturn->JA		= JA;
	csrReturn->IA		= IA;
	csrReturn->nnz		= nnz;

    // Start processing based on data type
	gdf_error status = GDF_SUCCESS;

    switch(dType) {
    	case gdf_dtype::GDF_INT8:
    		status = runConverter<int8_t>(gdfData, csrReturn, offsets);
    	    break;
    	case gdf_dtype::GDF_INT16:
    		status = runConverter<int16_t>(gdfData, csrReturn, offsets);
    	    break;
    	case gdf_dtype::GDF_INT32:
    		status = runConverter<int32_t>(gdfData, csrReturn, offsets);
    	    break;
    	case gdf_dtype::GDF_INT64:
    		status = runConverter<int64_t>(gdfData, csrReturn, offsets);
    	    break;
    	case gdf_dtype::GDF_FLOAT32:
    		status = runConverter<float>(gdfData, csrReturn, offsets);
    	    break;
    	case gdf_dtype::GDF_FLOAT64:
    		status = runConverter<double>(gdfData, csrReturn, offsets);
    	    break;
    	default:
    		RMM_TRY(RMM_FREE(IA, 0));
    		RMM_TRY(RMM_FREE(JA, 0));
    		RMM_TRY(RMM_FREE(offsets, 0));
    		return GDF_UNSUPPORTED_DTYPE;
    }

    RMM_TRY(RMM_FREE(offsets, 0));

	return status;
}


template<typename T>
gdf_error runConverter(gdf_column **gdfData, csr_gdf *csrReturn, cudf::size_type * offsets) {

	cudf::size_type	numCols		= csrReturn->cols;
	cudf::size_type	numRows		= csrReturn->rows;

	//-- threads and blocks
    int threads = 1024;

    if ( numRows < 100 ) {
    	threads = 64;
    } else if (numRows < 256) {
    	threads = 128;
    } else  if ( numRows < 512) {
    	threads = 256;
    } else if ( numRows < 1024) {
    	threads = 512;
    }

    int blocks  = (numRows + threads - 1) / threads;

	T *			A;
	RMM_TRY(RMM_ALLOC((void**)&A, (sizeof(T) * csrReturn->nnz), 0));
	CUDA_TRY(cudaMemset(A, 0, (sizeof(T) * csrReturn->nnz)));

    // Now start moving the data and creating the CSR
	for ( cudf::size_type colId = 0; colId < numCols; colId++ ) {
		gdf_column *gdf = gdfData[colId];

		cudaCreateCSR<T><<<blocks, threads>>>(gdf->data, gdf->valid, gdf->dtype, colId, A, csrReturn->JA, offsets, numRows);

		CHECK_CUDA(0);
	}

	csrReturn->A = A;

	return gdf_error::GDF_SUCCESS;
}


/*
 * Move data over into CSR and possible convert format
 */
template<typename T>
__global__ void cudaCreateCSR(
		void *data, cudf::valid_type *valid, gdf_dtype dtype, int colId,
		T *A, int64_t *JA, cudf::size_type *offsets, cudf::size_type numRows)
{
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);			// get the tread ID which is also the row number

	if ( tid >= numRows)
		return;

	int bitmapIdx 	= whichBitmapCSR(tid);  						// which bitmap
	int bitIdx		= whichBitCSR(tid);								// which bit - over an 8-bit index

	cudf::valid_type	bitmap	= valid[bitmapIdx];

	if ( checkBitCSR( bitmap, bitIdx) ) {

		cudf::size_type offsetIdx	= offsets[tid];							// where should this thread start writing data

		A[offsetIdx]  = convertDataElement<T>(data, tid, dtype);
		JA[offsetIdx] = colId;

		++offsets[tid];
	}

}


/*
 * Compute the number of valid entries per rows - a row spans multiple gdf_colums -
 * There is one thread running per row, so just compute the sum for this row.
 *
 * the number of elements a valid array is actually ceil(numRows / 8) since it is a bitmap.  the total number of bits checked is equal to numRows
 *
 */
__global__ void determineValidRecCount(cudf::valid_type *valid, cudf::size_type numRows, cudf::size_type numCol, cudf::size_type * offset) {

	int tid = threadIdx.x + (blockDim.x * blockIdx.x);				// get the tread ID which is also the row number

	if ( tid >= numRows)
		return;

	int bitmapIdx 	= whichBitmapCSR(tid);  						// want the floor of the divide
	int bitIdx		= whichBitCSR(tid);								// which bit - over an 8-bit index

	cudf::valid_type	bitmap	= valid[bitmapIdx];

	if (checkBitCSR( bitmap, bitIdx) )
		++offset[tid];
}


/**
 * Convert the data element into a common format
 */
template<typename T>
__device__ T convertDataElement(void *data, int tid, gdf_dtype dtype) {

	T answer;

    switch(dtype) {
    	case gdf_dtype::GDF_INT8: {
    		int8_t *a = (int8_t *)data;
    		answer = (T)(a[tid]);
    		break;
    	}
    	case gdf_dtype::GDF_INT16: {
    		int16_t *b = (int16_t *)data;
    		answer = (T)(b[tid]);
    		break;
    	}
    	case gdf_dtype::GDF_INT32: {
    		int32_t *c = (int32_t *)data;
    		answer = (T)(c[tid]);
    		break;
    	}
    	case gdf_dtype::GDF_INT64: {
    		int64_t *d = (int64_t *)data;
    		answer = (T)(d[tid]);
    		break;
    	}
    	case gdf_dtype::GDF_FLOAT32: {
    		float *e = (float *)data;
    		answer = (T)(e[tid]);
    		break;
    	}
    	case gdf_dtype::GDF_FLOAT64: {
    		double *f = (double *)data;
    		answer = (T)(f[tid]);
    		break;
    	}
    }

    return answer;
}



