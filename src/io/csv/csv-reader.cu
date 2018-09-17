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

/*
 * The code  uses the Thrust library
 */

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <thrust/host_vector.h>

#include "type_conversion.cuh"

#include <gdf/gdf.h>
#include <gdf/errorutils.h>
 
#include "gdf/gdf_io.h"


constexpr int32_t HASH_SEED = 33;

using namespace std;

//-- define the structure for raw data handling - for internal use
typedef struct raw_csv_ {
    char *				data;			// on-device: the raw unprocessed CSV data - loaded as a large char * array
    uint64_t *			rec_bits;		// on-device: bitmap indicator of there a record break is located
    uint64_t *			field_bits;		// on-device: bitmap of where a field break is located (delimiter)
    int	*				recPerChunck;	// on-device: Number of records per bitmap chunks
    long * 				offsets;		// on-device: index of record - for multiple records per chunks it is starting index

    long*				d_num_records;	// on-device: Number of records.
    long*				recStart;		// on-device: Starting position of the records.

    char				delimiter;		// host: the delimiter
    char				terminator;		// host: the line terminator

    long				num_bytes;		// host: the number of bytes in the data
    long				num_bits;		// host: the number of 64-bit bitmaps (different than valid)
	long            	num_records;  	// host: number of records (per column)
	int					num_cols;		// host: number of columns
    vector<gdf_dtype>	dtypes;			// host: array of dtypes (since gdf_columns are not created until end)
    vector<string>		col_names;		// host: array of column names
} raw_csv_t;

//-- define the fields
typedef struct fields_info_ {
	int *				rec_id;			// on-device: the record index
	int *				col_id;			// on-device: the column index
	int *				start_idx;		// on-device: the starting bytes of the field
	int *				end_idx;		// on-device: the ending byte of the field - this could be the delimiter or eol
}  fields_info_t;

using string_pair = std::pair<const char*,size_t>;

//
//---------------create and process ---------------------------------------------
//
gdf_error parseArguments(csv_read_arg *args, raw_csv_t *csv);

gdf_error getColNamesAndTypes(const char **col_names, const  char **dtypes, raw_csv_t *d);

gdf_error updateRawCsv( const char * data, long num_bytes, raw_csv_t * csvData );
gdf_error allocateGdfDataSpace(gdf_column *);

gdf_dtype convertStringToDtype(std::string &dtype);


gdf_error freeCSVSpace(raw_csv_t * raw_csv);

gdf_error freeCsvData(char *data);

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { cerr << "ERROR:  " << error <<  "  in "  << txt << endl;  return error; }


template<typename T> 
gdf_error allocateTypeN(void *gpu, long N);

//
//---------------CUDA Kernel ---------------------------------------------
//
gdf_error launch_determineRecAndFields(raw_csv_t * data);  // breaks out fields and computed blocks - make main code cleaner

__global__ void determineRecAndFields(char *data, uint64_t * r_bits, uint64_t *f_bits, const char delim, long num_bytes, long num_bits, int * rec_count);


__device__ int findSetBit(int tid, long num_bits, uint64_t *f_bits, int x);


gdf_error launch_countRecords(raw_csv_t * csvData);
gdf_error launch_storeRecordStart(raw_csv_t * csvData);

gdf_error launch_dataConvertColumnsNew(raw_csv_t * raw_csv, void** d_gdf,  gdf_valid_type** valid, gdf_dtype* d_dtypes, string_pair	**str_cols, int row_offset);


__global__ void countRecords(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records);
__global__ void storeRecordStart(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records,long* recStart) ;

__global__ void convertCsvToGdfNew(char * raw_csv,char delim,long  num_records, int  num_columns,long* recStart,gdf_dtype* dtype,void** gdf_data,gdf_valid_type **valid,string_pair	**str_cols,int row_offset);

__device__ void removePrePostWhiteSpaces2(char *data, long* start_idx, long* end_idx);

//
//---------------CUDA Valid (8 blocks of 8-bits) Bitmap Kernels ---------------------------------------------
//
__device__ int whichBitmap(int record) { return (record/8);  }
__device__ int whichBit(int bit) { return (bit % 8);  }


__inline__ __device__ void validAtomicOR(gdf_valid_type* address, gdf_valid_type val)
{
	int32_t *base_address = (int32_t*)((gdf_valid_type*)address - ((size_t)address & 3));	
	int32_t int_val = (int32_t)val << (((size_t) address & 3) * 8);
	atomicOr(base_address, int_val);
}


__device__ void setBit(gdf_valid_type* address, int bit) {
	gdf_valid_type bitMask[8] 		= {1, 2, 4, 8, 16, 32, 64, 128};
	validAtomicOR(address, bitMask[bit]);
}


//
//---------------Debug stuff (can be deleted) ---------------------------------------------
//
void printCheck(raw_csv_t * csvData, int start_idx, int num_blocks, const char * text);
void printGdfCheck(gdf_column * gdf, int start_data_idx, int num_records, const char * text);
void printResults(gdf_column ** data, int num_col, int start_data_idx, int num_records);
void printInfoCheck(fields_info_t *info, int num_records, const char * text, int start_idx = 0);
void printStartPositions(raw_csv_t * csvData, int num_records);


/**
 * main entry point
 */
gdf_error read_csv(csv_read_arg *args)
{
	gdf_error error = gdf_error::GDF_SUCCESS;

	//-----------------------------------------------------------------------------
	// create the CSV data structure - this will be filled in as the CSV data is processed.
	// Done first to validate data types
	raw_csv_t * raw_csv = new raw_csv_t;
	error = parseArguments(args, raw_csv);
	checkError(error, "Call to parseArguments");

	// cout << "Num Column = " << args->num_cols << std::endl;
	// cout << "Delimiter  = " << raw_csv->delimiter << std::endl;
	// cout << "Terminator = " << raw_csv->terminator << std::endl;

	//-----------------------------------------------------------------------------
	// memory map in the data
	void * 			map_data = NULL;
	struct stat     st;
	int				fd;

	fd = open(args->file_path, O_RDONLY );

	if (fd < 0) 		{ close(fd); checkError(GDF_FILE_ERROR, "Error opening file"); }
	if (fstat(fd, &st)) { close(fd); checkError(GDF_FILE_ERROR, "cannot stat file");   }

	raw_csv->num_bytes = st.st_size;

	map_data = mmap(0, raw_csv->num_bytes, PROT_READ, MAP_PRIVATE, fd, 0);

    if (map_data == MAP_FAILED) { close(fd); checkError(GDF_C_ERROR, "Error mapping file"); }

	// cout << "Num Column = " << args->num_cols << std::endl;
	// cout << "Bytes Read = " << raw_csv->num_bytes << std::endl;

	// if ( raw_csv->num_bytes < 500)  cout << "Data = \n"     << ((const char *)map_data) << endl;


	//-----------------------------------------------------------------------------
	//---  create a structure to hold variables used to parse the CSV data
	error = updateRawCsv( (const char *)map_data, (long)raw_csv->num_bytes, raw_csv );
	checkError(error, "call to createRawCsv");

	//-----------------------------------------------------------------------------
	//---  done with host data
	close(fd);
	munmap(map_data, raw_csv->num_bytes);


	//-----------------------------------------------------------------------------
	// find the record and fields points (in bitmaps)
	cudaDeviceSynchronize();
	error = launch_countRecords(raw_csv);
	checkError(error, "call to record counter");

	// cout << "The number of rows detected was " << raw_csv->num_records << endl;

	//-----------------------------------------------------------------------------
	//-- Allocate space to hold the record starting point

	CUDA_TRY( cudaMallocManaged ((void**)&raw_csv->recStart,(sizeof(long) * (raw_csv->num_records + 1))) );
	CUDA_TRY( cudaMemset(raw_csv->recStart, 		0, 		(sizeof(long) * (raw_csv->num_records + 1))) );
	CUDA_TRY( cudaMemset(raw_csv->d_num_records,	0, 		(sizeof(long) )) ) ;

	//-----------------------------------------------------------------------------
	//-- Scan data and set the starting positions
	error = launch_storeRecordStart(raw_csv);
	checkError(error, "call to record initial position store");

	cudaDeviceSynchronize();

	thrust::sort(raw_csv->recStart, raw_csv->recStart + raw_csv->num_records + 1);

	raw_csv->num_records -= (args->skiprows + args->skipfooter); 

	//-----------------------------------------------------------------------------
	// free up space that is no longer needed
	error = freeCSVSpace(raw_csv);
	checkError(error, "freeing raw_csv_t space");

	//--- allocate space for the results
	gdf_column **cols = (gdf_column **)malloc( sizeof(gdf_column *) * raw_csv->num_cols);

	void **d_data;
	gdf_valid_type **d_valid;
	CUDA_TRY( cudaMallocManaged ((void**)&d_data, 		(sizeof(void *)		* raw_csv->num_cols)) );
	CUDA_TRY( cudaMallocManaged ((void**)&d_valid, 		(sizeof(gdf_valid_type *)		* raw_csv->num_cols)) );

	gdf_dtype* d_dtypes;
	CUDA_TRY( cudaMallocManaged ((void**)&d_dtypes, sizeof(gdf_dtype) * (raw_csv->num_cols)) );

	int stringColCount=0;
	for (int col = 0; col < raw_csv->num_cols; col++) {
		if(raw_csv->dtypes[col]==gdf_dtype::GDF_STRING)
			stringColCount++;
	}

	string_pair** str_cols ;
	cudaMallocManaged ((void**)&str_cols, 		(sizeof(string_pair *)		* stringColCount));

	for (int col = 0; col < stringColCount; col++) {
		cudaMallocManaged ((void**)(str_cols + col), sizeof(string_pair) * (raw_csv->num_records));
	}


	for (int col = 0; col < raw_csv->num_cols; col++) {

		gdf_column *gdf = (gdf_column *)malloc(sizeof(gdf_column) * 1);

		gdf->size		= raw_csv->num_records;
		gdf->dtype		= raw_csv->dtypes[col];
		gdf->null_count	= 0;						// will be filled in later

		//--- column name
		std::string str = raw_csv->col_names[col];
		int len = str.length() + 1;
		gdf->col_name = (char *)malloc(sizeof(char) * len);
		memcpy(gdf->col_name, str.c_str(), len);
		gdf->col_name[len -1] = '\0';

		allocateGdfDataSpace(gdf);

		cols[col] 		= gdf;
		d_dtypes[col] 	= raw_csv->dtypes[col];
		d_data[col] 	= gdf->data;
		d_valid[col] 	= gdf->valid;
	}


	launch_dataConvertColumnsNew(raw_csv,d_data, d_valid, d_dtypes,str_cols, args->skiprows);

	for (int col = 0; col < stringColCount; col++) {
		//  TO-DO:  get a string class
		// d_data[col] = (void*) new Strings(str_cols[col],size_t(raw_csv->num_records));
	}

	cudaFree(d_data);

	error = freeCsvData(raw_csv->data);
	checkError(error, "call to cudaFree(raw_csv->data)" );
	delete raw_csv;

	//printResults(cols, raw_csv->num_cols, (raw_csv->num_records - 101), 100);

	args->data 			= cols;
	args->num_cols_out	= raw_csv->num_cols;
	args->num_rows_out	= raw_csv->num_records;

	return error;
}

//------------------------------------------------------------------------------------------------------------------------------



/**
 * This creates the basic gdf_coulmn structure
 *
 */
gdf_error getColNamesAndTypes(const char **col_names, const  char **dtypes, raw_csv_t *d)
{

	// start creating space for each column
	for ( int x = 0; x < d->num_cols; x++) {

		std::string col_name 	= col_names[x];
		std::string temp_type 	= dtypes[x];
		gdf_dtype col_dtype		= convertStringToDtype( temp_type );

		if (col_dtype == GDF_invalid)
			return GDF_UNSUPPORTED_DTYPE;

		d->dtypes.push_back(col_dtype);
		d->col_names.push_back(col_name);
	}

	return gdf_error::GDF_SUCCESS;
}


gdf_error parseArguments(csv_read_arg *args, raw_csv_t *csv)
{
	//--- For the initial version, the number of columns need to be specified
	csv->num_cols		= args->num_cols;
	csv->num_records	= 0;


	//----- Delimiter
	if ( args->delim_whitespace == true) {
		csv->delimiter = ' ';
	} else {
		csv->delimiter = args->delimiter;
	}

	//----- Line Delimiter
	csv->terminator = args->lineterminator;


	//--- Now look at column name and
	for ( int x = 0; x < csv->num_cols; x++) {

		std::string col_name 	= args->names[x];
		std::string temp_type 	= args->dtype[x];
		gdf_dtype col_dtype		= convertStringToDtype( temp_type );

		if (col_dtype == GDF_invalid)
			return GDF_UNSUPPORTED_DTYPE;

		csv->dtypes.push_back(col_dtype);
		csv->col_names.push_back(col_name);
	}

	return gdf_error::GDF_SUCCESS;
}

/*
 * What is passed in is the data type as a string, need to convert that into gdf_dtype enum
 */
gdf_dtype convertStringToDtype(std::string &dtype) {


	if (dtype.compare( "str") == 0) 		return GDF_CATEGORY;
	if (dtype.compare( "date") == 0) 		return GDF_DATE64;
	if (dtype.compare( "category") == 0) 	return GDF_CATEGORY;
	if (dtype.compare( "float") == 0)		return GDF_FLOAT32;
	if (dtype.compare( "float32") == 0)		return GDF_FLOAT32;
	if (dtype.compare( "float64") == 0)		return GDF_FLOAT64;
	if (dtype.compare( "double") == 0)		return GDF_FLOAT64;
	if (dtype.compare( "short") == 0)		return GDF_INT16;
	if (dtype.compare( "int") == 0)			return GDF_INT32;
	if (dtype.compare( "int32") == 0)		return GDF_INT32;
	if (dtype.compare( "int64") == 0)		return GDF_INT64;
	if (dtype.compare( "long") == 0)		return GDF_INT64;

	return GDF_invalid;
}


/*
 * Create the raw_csv_t structure and allocate space on the GPU
 */
gdf_error updateRawCsv( const char * data, long num_bytes, raw_csv_t * raw ) {

	int num_bits = (num_bytes + 63) / 64;

	CUDA_TRY(cudaMallocManaged ((void**)&raw->data, 		(sizeof(char)		* num_bytes)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->rec_bits, 	(sizeof(uint64_t)	* num_bits)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->field_bits, 	(sizeof(uint64_t)	* num_bits)));
	CUDA_TRY(cudaMallocManaged ((void**)&raw->recPerChunck,	(sizeof(int) 		* num_bits)) );
	CUDA_TRY(cudaMallocManaged ((void**)&raw->offsets, 		((sizeof(long)		* num_bits) + 2)) );

	CUDA_TRY(cudaMallocManaged ((void**)&raw->d_num_records, sizeof(long)) );

	CUDA_TRY(cudaMemcpy(raw->data, data, num_bytes, cudaMemcpyHostToDevice));
	
	CUDA_TRY( cudaMemset(raw->rec_bits, 	0, (sizeof(uint64_t) 	* num_bits)) );
	CUDA_TRY( cudaMemset(raw->field_bits, 	0, (sizeof(uint64_t) 	* num_bits)) );
	CUDA_TRY( cudaMemset(raw->recPerChunck, 0, (sizeof(int) 		* num_bits)) );
	CUDA_TRY( cudaMemset(raw->offsets, 		0, ((sizeof(long) 		* num_bits) + 2)) );

	CUDA_TRY( cudaMemset(raw->d_num_records,0, ((sizeof(long)) )) );

	raw->num_bits  = num_bits;

	return GDF_SUCCESS;
}


/*
 * For each of the gdf_cvolumns, create the on-device space.  the on-host fields should already be filled in
 */
gdf_error allocateGdfDataSpace(gdf_column *gdf) {

	long N = gdf->size;
	int num_bitmaps = (N + 7) / 8;			// 8 bytes per bitmap

	//--- allocate space for the valid bitmaps
	CUDA_TRY(cudaMallocManaged(&gdf->valid, (sizeof(gdf_valid_type) 	* num_bitmaps)));
	CUDA_TRY(cudaMemset(gdf->valid, 0, (sizeof(gdf_valid_type) 	* num_bitmaps)) );

	//--- Allocate space for the data
	switch(gdf->dtype) {
		case gdf_dtype::GDF_INT8:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int8_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int8_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT16:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int16_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int16_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT32:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int32_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int32_t) 	* N)) );
			break;
		case gdf_dtype::GDF_INT64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(int64_t) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(int64_t) 	* N)) );
			break;
		case gdf_dtype::GDF_FLOAT32:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(float) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(float) 	* N)) );
			break;
		case gdf_dtype::GDF_FLOAT64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(double) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(double) 	* N)) );
			break;
		case gdf_dtype::GDF_DATE64:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(gdf_date64) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(gdf_date64) 	* N)) );
			break;
		case gdf_dtype::GDF_CATEGORY:
			CUDA_TRY(cudaMallocManaged(&gdf->data, (sizeof(gdf_category) * N)));
			CUDA_TRY(cudaMemset(gdf->data, 0, (sizeof(gdf_category) 	* N)) );
			break;
		case gdf_dtype::GDF_STRING:
			// Memory for gdf->data allocated by string class eventually
			break;
		default:
			return GDF_UNSUPPORTED_DTYPE;
	}

	return gdf_error::GDF_SUCCESS;
}


gdf_error freeCSVSpace(raw_csv_t * raw_csv)
{
	CUDA_TRY(cudaFree(raw_csv->rec_bits));
	CUDA_TRY(cudaFree(raw_csv->field_bits));
	CUDA_TRY(cudaFree(raw_csv->recPerChunck));
	CUDA_TRY(cudaFree(raw_csv->offsets));

	return gdf_error::GDF_SUCCESS;
}


gdf_error freeCsvData(char *data)
{
	CUDA_TRY(cudaFree(data));

	return gdf_error::GDF_SUCCESS;

}

//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------
gdf_error launch_determineRecAndFields(raw_csv_t * csvData) {

	char 		*data 		= csvData->data;
	uint64_t 	*r_bits		= csvData->rec_bits;
	uint64_t 	*f_bits		= csvData->field_bits;
	long 		num_bytes	= csvData->num_bytes;
	int			*rec_count	= csvData->recPerChunck;
	long 		numBitmaps 	= csvData->num_bits;
	char		delim		= csvData->delimiter;


	/*
	 * Each bi map is for a 64-byte chunk, and technically we could do a thread per 64-bytes.
	 * However, that doesn't seem efficient.

	 *      Note: could do one thread per byte, but that would require a lock on the bit map
	 *
	 */
	int threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	int blocks = (numBitmaps + (threads -1)) / threads ;

	determineRecAndFields <<< blocks, threads >>> (data, r_bits, f_bits, delim, num_bytes, numBitmaps, rec_count);

	CUDA_TRY(cudaGetLastError());
	return GDF_SUCCESS;
}


__global__ void determineRecAndFields(char *data, uint64_t * r_bits, uint64_t *f_bits, const char delim, long num_bytes, long num_bits, int * rec_count) {

	// thread IDs range per block, so also need the block id
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID - multiple of 64
	long did = tid * 64;

	char *raw = (data + did);

	int byteToProcess = ((did + 64) < num_bytes) ? 64 : (num_bytes - did);
	uint64_t r_bits_local = 0;

	// process the data
	int x = 0;
	for (x = 0; x < byteToProcess; x++) {

		// fields
		if (raw[x] == delim) {
			f_bits[tid] |= 1UL << x;
		} else {
			// records
			if (raw[x] == '\n') {
				r_bits_local |= 1UL << x;

			}	else if (raw[x] == '\r' && raw[x +1] == '\n') {
				x++;
				r_bits_local |= 1UL << x;
			}
		}
	}

	// save the number of records detected within this block
	uint64_t bitmap = r_bits_local;
	int rec_count_local = 0;
	while (bitmap)
	{
		rec_count_local += bitmap & 1;
		bitmap >>= 1;
	 }

	if ( tid == 0 )
		++rec_count_local;

	rec_count[tid] = rec_count_local;
	r_bits[tid] = r_bits_local;
}

gdf_error launch_countRecords(raw_csv_t * csvData) {

	char 		*data 		= csvData->data;
	long 		num_bytes	= csvData->num_bytes;
	long 		numBitmaps 	= csvData->num_bits;
	char		delim		= csvData->delimiter;
	long 		*d_num_records = csvData->d_num_records;
	char		terminator	= csvData->terminator;

	/*
	 * Each bitmap is for a 64-byte chunk,
	 *
	 *  Note: could do one thread per byte, but that would require a lock on the bit map
	 *
	 */
	int64_t threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	int64_t blocks = (numBitmaps + (threads -1)) / threads ;

	countRecords <<< blocks, threads >>> (data, delim, terminator, num_bytes, numBitmaps, d_num_records);

	CUDA_TRY(cudaGetLastError());

	long recs=-1;
	CUDA_TRY(cudaMemcpy(&recs, d_num_records, sizeof(long), cudaMemcpyDeviceToHost));
	csvData->num_records=recs;

	CUDA_TRY(cudaGetLastError());

	return GDF_SUCCESS;
}


__global__ void countRecords(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records) {

	// thread IDs range per block, so also need the block id
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID is a multiple of 64
	long did = tid * 64;

	char *raw = (data + did);

	int byteToProcess = ((did + 64) < num_bytes) ? 64 : (num_bytes - did);

	// process the data
	int x = 0;
	long newLinesFound=0;
	for (x = 0; x < byteToProcess; x++) {

		// records
		if (raw[x] == terminator) {
			newLinesFound++;
		}	else if (raw[x] == '\r' && raw[x +1] == '\n') {
			x++;
			newLinesFound++;
		}

	}
	atomicAdd((unsigned long long int*)num_records,(unsigned long long int)newLinesFound);
}


gdf_error launch_storeRecordStart(raw_csv_t * csvData) {

	char 		*data 		= csvData->data;
	long 		num_bytes	= csvData->num_bytes;
	long 		numBitmaps 	= csvData->num_bits;
	char		delim		= csvData->delimiter;
	char 		terminator	= csvData->terminator;

	long 		*d_num_records 	= csvData->d_num_records;
	long 		*recStart 		= csvData->recStart;

	/*
	 * Each bitmap is for a 64-byte chunk
	 *  Note: could do one thread per byte, but that would require a lock on the bit map
	 */
	int threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	int blocks = (numBitmaps + (threads -1)) / threads ;

	storeRecordStart <<< blocks, threads >>> (data, delim, terminator, num_bytes, numBitmaps,d_num_records,recStart);

	CUDA_TRY(cudaGetLastError());
	return GDF_SUCCESS;
}


__global__ void storeRecordStart(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records,long* recStart) {

	// thread IDs range per block, so also need the block id
	int tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID - multiple of 64
	long did = tid * 64;

	char *raw = (data + did);

	int byteToProcess = ((did + 64) < num_bytes) ? 64 : (num_bytes - did);

	if(tid==0){
		long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
		recStart[pos]=did+0;
	}

	// process the data
	int x = 0;
	for (x = 0; x < byteToProcess; x++) {

		// records
		if (raw[x] == terminator) {

			long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
			recStart[pos]=did+x+1;

		}	else if (raw[x] == '\r' && raw[x +1] == '\n') {

			x++;
			long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
			recStart[pos]=did+x+1;
		}

	}
}


//----------------------------------------------------------------------------------------------------------------



gdf_error launch_dataConvertColumnsNew(raw_csv_t * raw_csv, void **gdf, gdf_valid_type** valid, gdf_dtype* d_dtypes,string_pair	**str_cols, int row_offset) {

	int64_t threads 	= 1024;
	int64_t blocks 		= (  raw_csv->num_records + (threads -1)) / threads ;


	convertCsvToGdfNew <<< blocks, threads >>>(
		raw_csv->data,
		raw_csv->delimiter,
		raw_csv->num_records,
		raw_csv->num_cols,
		raw_csv->recStart,
		d_dtypes,
		gdf,
		valid,
		str_cols,
		row_offset
	);


	return GDF_SUCCESS;
}


/*
 * Data is processed in 64-bytes chunks - so the number of total threads (tid) is equal to the number of bitmaps
 * a thread will start processing at the start of a record and end of the end of a record, even if it crosses
 * a 64-byte boundary
 *
 * tid = 64-byte index
 * did = offset into character data
 * offset = record index (starting if more that 1)
 *
 */
__global__ void convertCsvToGdfNew(
		char 			*raw_csv,
		char 			delim,
		long  			num_records,
		int  			num_columns,
		long 			*recStart,
		gdf_dtype 		*dtype,
		void			**gdf_data,
		gdf_valid_type 	**valid,
		string_pair	**str_cols,
		int row_offset)
{



	// thread IDs range per block, so also need the block id
	long	rec_id  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( rec_id >= num_records)
		return;

	long start 		= recStart[rec_id + row_offset];
	long stop 		= recStart[rec_id + 1 + row_offset]-1;
	long pos 		= start;
	int  col 		= 0;
	int  stringCol 	= 0;
	while(col<num_columns){

		if(start>stop)
			break;

		while(true){
			if(raw_csv[pos]==delim){
				break;
			}
			if(pos>=stop)
				break;
			pos++;
		}

		// if(threadIdx.x==0&&blockIdx.x==0){
		// 	// printf("^^^^ %d %ld %ld \n", col, start, pos);	
		// }
		long tempPos=pos-1;

		// if(blockIdx.x == 0 && threadIdx.x==0){
			if(dtype[col] != gdf_dtype::GDF_CATEGORY && dtype[col] != gdf_dtype::GDF_STRING){
				removePrePostWhiteSpaces2(raw_csv, &start, &tempPos);
			}
		// }


		// if((start<=(pos-1) && dtype[col]!=gdf_dtype::GDF_STRING) || (start==pos && dtype[col]==gdf_dtype::GDF_STRING)) { // Empty strings are consider legal values
		if(start<=(tempPos)) { // Empty strings are not legal values

			switch(dtype[col]) {
				case gdf_dtype::GDF_INT8:
				{
					int8_t *gdf_out = (int8_t *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoInt<int8_t>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_INT16: {
					int16_t *gdf_out = (int16_t *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoInt<int16_t>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_INT32:
				{
					int32_t *gdf_out = (int32_t *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoInt<int32_t>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_INT64:
				{
					int64_t *gdf_out = (int64_t *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoInt<int64_t>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_FLOAT32:
				{
					float *gdf_out = (float *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoFloat<float>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_FLOAT64:
				{
					double *gdf_out = (double *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoFloat<double>(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_DATE64:
				{
					gdf_date64 *gdf_out = (gdf_date64 *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoDate(raw_csv, start, tempPos);
				}
					break;
				case gdf_dtype::GDF_CATEGORY:
				{
					gdf_category *gdf_out = (gdf_category *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoHash(raw_csv, start, pos, HASH_SEED);
				}
					break;
				case gdf_dtype::GDF_STRING:{
					str_cols[stringCol][rec_id].first 	= raw_csv+start;
					str_cols[stringCol][rec_id].second 	= size_t(pos-start);
					stringCol++;
				}
					break;
				default:
					break;
			}

			// if (dtype[col]!=gdf_dtype::GDF_STRING){		
				// set the valid bitmap - all bits were set to 0 to start
				int bitmapIdx 	= whichBitmap(rec_id + col);  	// which bitmap
				int bitIdx		= whichBit(rec_id + col);		// which bit - over an 8-bit index
				setBit(valid[col]+bitmapIdx, bitIdx);		// This is done with atomics

			// }
		}
		else if(dtype[col]==gdf_dtype::GDF_STRING){
			str_cols[stringCol][rec_id].first 	= NULL;
			str_cols[stringCol][rec_id].second 	= 0;
			stringCol++;
			if(threadIdx.x==0&&blockIdx.x==0){
				printf("^^^^ %d %ld 0 \n", col, start);	
			}

		}

		pos++;
		start=pos;
		col++;	
	}
}

__device__
void removePrePostWhiteSpaces2(char *data, long* start_idx, long* end_idx) {
	while(*start_idx < *end_idx && data[*start_idx] == ' ')
		*start_idx=*start_idx+1;
	while(*start_idx < *end_idx && data[*end_idx] == ' ')
		*end_idx=*end_idx-1;
}


/*
 * Return which bit is set
 * x is the occurrence: 1 = first, 2 = seconds, ...
 */
__device__ int findSetBit(int tid, long num_bits, uint64_t *r_bits, int x) {

	int idx = tid;

	if ( x == 0 )
		return -1;

	int withinBitCount = 0;
	int offset = 0;
	int found  = 0;

	uint64_t bitmap = r_bits[idx];

	while (found != x)
	{
		if(bitmap == 0)
		{
			idx++;
			if (idx >= num_bits)
				return -1;
			bitmap = r_bits[idx];
			offset += 64;
			withinBitCount = 0;
		}

		if ( bitmap & 1 ) {
			found++;			//found a set bit
		}

		bitmap >>= 1;
		++withinBitCount;
	 }

	offset += withinBitCount -1;


	return offset;
}





//---------------------------------------------------------------------------------------------------------------
//
//			Debug functions below this point
//
//---------------------------------------------------------------------------------------------------------------

void printCheck(raw_csv_t * csvData, int start_data_idx, int num_blocks, const char * text) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " 		<< text << std::endl;
	std::cout << "\tNumber of Bytes   - " << csvData->num_bytes 	<< std::endl;
	std::cout << "\tNumber of Bitmaps - " << csvData->num_bits  	<< std::endl;
	std::cout << "\tNumber of Records - " << csvData->num_records  	<< std::endl;


	char * data = csvData->data;

	int data_idx 	= start_data_idx;

	if ( data_idx != 0)
		while ( data_idx % 64 != 0 )
			data_idx++;

	std::cout << "\tStarting Index specified - " << start_data_idx << "  adjusted to " << data_idx << std::endl;

	int bitmap_idx  = data_idx / 64;

	std::cout << "\tStarting Bitmap Index - " << bitmap_idx  << std::endl;
	std::cout << "[data, bit] =>                 64 bytes of data                 \t rec_bit   field_bits    Record counts"  << std::endl;


	for ( int loop = 0; loop < num_blocks; loop++) {
		std::cout << "[" << std::setw(6) << data_idx << " ,  " << std::setw(6) << bitmap_idx << "] =>  ";

		 for ( int x = 0; x < 64; x++) {

			if ( data_idx < csvData->num_bytes) {
				if (data[data_idx] == '\n')
					std::cout << "\033[1m\033[31mNL\033[0m ";
				else
					std::cout << data[data_idx] << " ";
			}

			++data_idx;
		 }

		std::cout << " =>  " << std::setw(25) << csvData->rec_bits[bitmap_idx];
		std::cout << "\t" << std::setw(25) << csvData->field_bits[bitmap_idx];
		std::cout << "\t" << std::setw(2) << csvData->recPerChunck[bitmap_idx];
		std::cout << "\t" << std::setw(2) << csvData->offsets[bitmap_idx];
		std::cout << std::endl;

		++bitmap_idx;
	}

	std::cout << "--------------------------------\n\n";

}


void printGdfCheck(gdf_column * gdf, int start_data_idx, int num_records, const char * text) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " << text << std::endl;
	std::cout << "\tCol: " << gdf->col_name  <<  " and data type of " << gdf->dtype << std::endl;

	for ( int x = 0; x < num_records; x++) {
		switch(gdf->dtype) {
			case gdf_dtype::GDF_INT8:
			{
				int8_t *gdf_out = (int8_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT16: {
				int16_t *gdf_out = (int16_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT32:
			{
				int32_t *gdf_out = (int32_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_INT64:
			{
				int64_t *gdf_out = (int64_t *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_FLOAT32:
			{
				float *gdf_out = (float *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_FLOAT64:
			{
				double *gdf_out = (double *)gdf->data;
				std::cout << "\tRec[ " << x << "] is "<< gdf_out[x] << std::endl;
			}
				break;
			case gdf_dtype::GDF_DATE64:
				break;
			case gdf_dtype::GDF_CATEGORY:
				break;
			case gdf_dtype::GDF_STRING:
				break;
			default:
				break;
		}
	}

	std::cout << "--------------------------------\n\n";

}


void printResults(gdf_column ** data, int num_col, int start_data_idx, int num_records) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Printing Results (dependent on Unified Memory) " << std::endl;

	for ( int c = 0;  c < num_col; c++)
	{
		gdf_column *gdf = data[c];

		std::cout << std::setw(2) << c << " =>  " << std::setw(50) << gdf->col_name << "\t" << gdf->dtype << "\t" << gdf->size << "\t" << gdf->null_count << std::endl;
	}
	std::cout << std::endl;

	long x = 0;
	for ( int i = 0; i < num_records; i++) {
		x = start_data_idx + i;

		std::cout << "\tRec[ " << x << "] is:  ";


		for ( int c = 0;  c < num_col; c++)
		{
			gdf_column *gdf = data[c];


			switch(gdf->dtype) {
				case gdf_dtype::GDF_INT8:
				{
					int8_t *gdf_out = (int8_t *)gdf->data;
					std::cout << "\t"  << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT16: {
					int16_t *gdf_out = (int16_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT32:
				{
					int32_t *gdf_out = (int32_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_INT64:
				{
					int64_t *gdf_out = (int64_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_FLOAT32:
				{
					float *gdf_out = (float *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_FLOAT64:
				{
					double *gdf_out = (double *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_DATE64:
				{
					int64_t *gdf_out = (int64_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
				break;
				case gdf_dtype::GDF_CATEGORY:
				{
					int32_t *gdf_out = (int32_t *)gdf->data;
					std::cout << "\t" << gdf_out[x];
				}
					break;
				case gdf_dtype::GDF_STRING:
					break;
				default:
					break;
			}
		}
		std::cout << std::endl;
	}

	std::cout << "--------------------------------\n\n";

}



void printInfoCheck(fields_info_t *info, int num_records, const char * text, int start_idx) {

	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Checking (dependent on Unified Memory) - " << text << std::endl;

	std::cout << "\tRec Id\tCol Id\tStart Idx\tEnd Idx" << std::endl;

	for ( int x = start_idx; x < num_records; x++) {
		std::cout << "\t" << info->rec_id[x] << "\t" << info->col_id[x] << "\t" << info->start_idx[x] << "\t" << info->end_idx[x]   << std::endl;
	}

}


void printStartPositions(raw_csv_t * raw_csv, int num_records)
{
	cudaDeviceSynchronize();

	std::cout << "\n--------------------------------\n";
	std::cout << "Print Start Indexes (dependent on Unified Memory) " << std::endl;

	for ( int x = 0; x < num_records; x++) {
		std::cout << "\t" <<  raw_csv->recStart[x] << std::endl;
	}




}




















