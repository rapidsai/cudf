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
 * @file csv-reader.cu  code to read csv data
 *
 * CSV Reader
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
#include "date-time-parser.cuh"

#include <gdf/gdf.h>
#include <gdf/errorutils.h>
 
#include "gdf/gdf_io.h"
#include "../../nvtx_utils.h"

#include "rmm.h"

constexpr int32_t HASH_SEED = 33;

using namespace std;

//-- define the structure for raw data handling - for internal use
typedef struct raw_csv_ {
    char *				data;			// on-device: the raw unprocessed CSV data - loaded as a large char * array
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

    bool				dayfirst;
} raw_csv_t;


using string_pair = std::pair<const char*,size_t>;

//
//---------------create and process ---------------------------------------------
//
gdf_error parseArguments(csv_read_arg *args, raw_csv_t *csv);
gdf_error getColNamesAndTypes(const char **col_names, const  char **dtypes, raw_csv_t *d);
gdf_error updateRawCsv( const char * data, long num_bytes, raw_csv_t * csvData );
gdf_error allocateGdfDataSpace(gdf_column *);
gdf_dtype convertStringToDtype(std::string &dtype);

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { cerr << "ERROR:  " << error <<  "  in "  << txt << endl;  return error; }

//
//---------------CUDA Kernel ---------------------------------------------
//

__device__ int findSetBit(int tid, long num_bits, uint64_t *f_bits, int x);

gdf_error launch_countRecords(raw_csv_t * csvData);
gdf_error launch_storeRecordStart(raw_csv_t * csvData);
gdf_error launch_dataConvertColumns(raw_csv_t * raw_csv, void** d_gdf,  gdf_valid_type** valid, gdf_dtype* d_dtypes, string_pair	**str_cols, int row_offset, long *);

__global__ void countRecords(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records);
__global__ void storeRecordStart(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records,long* recStart) ;

__global__ void convertCsvToGdf(char *csv,char delim, long num_records, int num_columns,long* recStart,gdf_dtype* dtype,void** gdf_data,gdf_valid_type **valid,string_pair **str_cols, int row_offset, bool, long *);

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



/**
 * @brief read in a CSV file
 *
 * Read in a CSV file, extract all fields, and return a GDF (array of gdf_columns)
 *
 * @param[in and out] args the input arguments, but this also contains the returned data
 *
 * Arguments:
 *
 *  Required Arguments
 * 		file_path			- 	file location to read from	- currently the file cannot be compressed
 * 		num_cols			-	number of columns in the names and dtype arrays
 * 		names				-	ordered List of column names, this is a required field
 * 		dtype				- 	ordered List of data types, this is required
 *
 * 	Optional
 * 		lineterminator		-	define the line terminator character.  Default is  '\n'
 * 		delimiter			-	define the field separator, default is ','   This argument is also called 'sep'
 * 		delim_whitespace	-	use white space as the delimiter - default is false.  This overrides the delimiter argument
 * 		skipinitialspace	-	skip white spaces after the delimiter - default is false
 *
 * 		skiprows			-	number of rows at the start of the files to skip, default is 0
 * 		skipfooter			-	number of rows at the bottom of the file to skip - default is 0
 *
 * 		dayfirst			-	is the first value the day?  DD/MM  versus MM/DD
 *
 *
 *  Output
 *  	num_cols_out		-	Out: return the number of columns read in
 *  	num_rows_out		- 	Out: return the number of rows read in
 *  	gdf_column		**data	-  Out: return the array of *gdf_columns
 *
 *
 * @return gdf_error
 *
 */
gdf_error read_csv(csv_read_arg *args)
{

	PUSH_RANGE("LIBGDF_READ_CSV",READ_CSV_COLOR);
	gdf_error error = gdf_error::GDF_SUCCESS;

	//-----------------------------------------------------------------------------
	// create the CSV data structure - this will be filled in as the CSV data is processed.
	// Done first to validate data types
	raw_csv_t * raw_csv = new raw_csv_t;
	error = parseArguments(args, raw_csv);
	checkError(error, "Call to parseArguments");

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

	//-----------------------------------------------------------------------------
	//-- Allocate space to hold the record starting point
	CUDA_TRY( cudaMallocManaged ((void**)&raw_csv->recStart,(sizeof(long) * (raw_csv->num_records + 1))) );
	CUDA_TRY( cudaMemset(raw_csv->d_num_records,	0, 		(sizeof(long) )) ) ;

	//-----------------------------------------------------------------------------
	//-- Scan data and set the starting positions
	error = launch_storeRecordStart(raw_csv);
	checkError(error, "call to record initial position store");

	cudaDeviceSynchronize();

	thrust::sort(thrust::device, raw_csv->recStart, (raw_csv->recStart + raw_csv->num_records + 1));

	raw_csv->num_records -= (args->skiprows + args->skipfooter); 

	//-----------------------------------------------------------------------------
	//--- allocate space for the results
	gdf_column **cols = (gdf_column **)malloc( sizeof(gdf_column *) * raw_csv->num_cols);

	void **d_data;
	gdf_valid_type **d_valid;
    long	*d_valid_count;

	CUDA_TRY( cudaMallocManaged ((void**)&d_data, 		(sizeof(void *)				* raw_csv->num_cols)) );
	CUDA_TRY( cudaMallocManaged ((void**)&d_valid, 		(sizeof(gdf_valid_type *)	* raw_csv->num_cols)) );
	CUDA_TRY( cudaMallocManaged ((void**)&d_valid_count,(sizeof(long) 				* raw_csv->num_cols)) );
	CUDA_TRY( cudaMemset(d_valid_count,	0, 				(sizeof(long) 				* raw_csv->num_cols)) );


	gdf_dtype* d_dtypes;
	CUDA_TRY( cudaMallocManaged ((void**)&d_dtypes, 	sizeof(gdf_dtype) 			* (raw_csv->num_cols)) );

	int stringColCount=0;
	for (int col = 0; col < raw_csv->num_cols; col++) {
		if(raw_csv->dtypes[col]==gdf_dtype::GDF_STRING)
			stringColCount++;
	}

	string_pair** str_cols = NULL;

	if (stringColCount > 0 ) {
		CUDA_TRY( cudaMallocManaged ((void**)&str_cols, 	(sizeof(string_pair *)		* stringColCount)) );

		for (int col = 0; col < stringColCount; col++) {
			CUDA_TRY( cudaMallocManaged ((void**)(str_cols + col), sizeof(string_pair) * (raw_csv->num_records)) );
		}
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

	launch_dataConvertColumns(raw_csv,d_data, d_valid, d_dtypes,str_cols, args->skiprows, d_valid_count);

	for (int col = 0; col < stringColCount; col++) {
		//  TO-DO:  get a string class
		CUDA_TRY( cudaFree (str_cols [col] ) );

	}

	//--- set the null count
	for ( int col = 0; col < raw_csv->num_cols; col++) {
		long x = 0;
		CUDA_TRY(cudaMemcpy(&x, &d_valid_count[col], sizeof(long), cudaMemcpyDeviceToHost));

		cols[col]->null_count = raw_csv->num_records - x;
	}


	// free up space that is no longer needed
	if (str_cols != NULL)
		CUDA_TRY( cudaFree (str_cols) );

	CUDA_TRY( cudaFree(d_valid) );
	CUDA_TRY( cudaFree(d_data) );
	CUDA_TRY( cudaFree(d_valid_count) );
	CUDA_TRY( cudaFree(raw_csv->recStart));
	CUDA_TRY( cudaFree(raw_csv->data));
	CUDA_TRY( cudaFree(raw_csv->d_num_records));

	delete raw_csv;

	args->data 			= cols;
	args->num_cols_out	= raw_csv->num_cols;
	args->num_rows_out	= raw_csv->num_records;

	POP_RANGE();
	return error;
}

//------------------------------------------------------------------------------------------------------------------------------



/*
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

	csv->dayfirst = args->dayfirst;

	return gdf_error::GDF_SUCCESS;
}

/*
 * What is passed in is the data type as a string, need to convert that into gdf_dtype enum
 */
gdf_dtype convertStringToDtype(std::string &dtype) {


	if (dtype.compare( "str") == 0) 		return GDF_CATEGORY;
	if (dtype.compare( "date") == 0) 		return GDF_DATE64;
	if (dtype.compare( "date32") == 0) 		return GDF_DATE32;
	if (dtype.compare( "date64") == 0) 		return GDF_DATE64;
	if (dtype.compare( "timestamp") == 0)	return GDF_TIMESTAMP;
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

	CUDA_TRY(cudaMallocManaged ((void**)&raw->d_num_records, sizeof(long)) );

	CUDA_TRY(cudaMemcpy(raw->data, data, num_bytes, cudaMemcpyHostToDevice));

	CUDA_TRY( cudaMemset(raw->d_num_records,0, ((sizeof(long)) )) );

	raw->num_bits  = num_bits;

	return GDF_SUCCESS;
}


/*
 * For each of the gdf_cvolumns, create the on-device space.  the on-host fields should already be filled in
 */
gdf_error allocateGdfDataSpace(gdf_column *gdf) {

	long N = gdf->size;
	long num_bitmaps = (N + 31) / 8;			// 8 bytes per bitmap

	//--- allocate space for the valid bitmaps
	RMM_TRY( rmmAlloc((void**)&gdf->valid, (sizeof(gdf_valid_type) 	* num_bitmaps), 0) );
	CUDA_TRY(cudaMemset(gdf->valid, 0, (sizeof(gdf_valid_type) 	* num_bitmaps)) );

	//--- Allocate space for the data
	int bytes_per_element = -1;
	gdf_error result = get_column_byte_width(gdf, &bytes_per_element);
	
	// TODO replace this once GDF_CATEGORY is added to get_column_byte_width
	if (GDF_UNSUPPORTED_DTYPE == result) 
	{ 
		if (gdf->dtype == GDF_CATEGORY) bytes_per_element = sizeof(gdf_category);
		else return result;
	}
	
	RMM_TRY( rmmAlloc((void**)&gdf->data, bytes_per_element * N, 0) );

	return gdf_error::GDF_SUCCESS;
}


//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------


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
	long tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID is a multiple of 64
	long did = tid * 64L;

	char *raw = (data + did);

	long byteToProcess = ((did + 64L) < num_bytes) ? 64L : (num_bytes - did);

	// process the data
	long x = 0;
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
	long threads 	= 1024;

	// Using the number of bitmaps as the size - data index is bitmap ID * 64
	long blocks = (numBitmaps + (threads -1)) / threads ;

	storeRecordStart <<< blocks, threads >>> (data, delim, terminator, num_bytes, numBitmaps,d_num_records,recStart);

	CUDA_TRY(cudaGetLastError());
	return GDF_SUCCESS;
}


__global__ void storeRecordStart(char *data, const char delim, const char terminator, long num_bytes, long num_bits, long* num_records,long* recStart) {

	// thread IDs range per block, so also need the block id
	long tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID - multiple of 64
	long did = tid * 64L;

	char *raw = (data + did);

	long byteToProcess = ((did + 64L) < num_bytes) ? 64L : (num_bytes - did);

	if(tid==0){
		long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
		recStart[pos]=did+0;
	}

	// process the data
	long x = 0;
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



gdf_error launch_dataConvertColumns(raw_csv_t * raw_csv, void **gdf, gdf_valid_type** valid, gdf_dtype* d_dtypes,string_pair **str_cols, int row_offset, long *num_valid) {

	int64_t threads 	= 1024;
	int64_t blocks 		= (  raw_csv->num_records + (threads -1)) / threads ;


	convertCsvToGdf <<< blocks, threads >>>(
		raw_csv->data,
		raw_csv->delimiter,
		raw_csv->num_records,
		raw_csv->num_cols,
		raw_csv->recStart,
		d_dtypes,
		gdf,
		valid,
		str_cols,
		row_offset,
		raw_csv->dayfirst,
		num_valid
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
__global__ void convertCsvToGdf(
		char 			*raw_csv,
		char 			delim,
		long  			num_records,
		int  			num_columns,
		long 			*recStart,
		gdf_dtype 		*dtype,
		void			**gdf_data,
		gdf_valid_type 	**valid,
		string_pair		**str_cols,
		int 			row_offset,
		bool			dayfirst,
		long			*num_valid
		)
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
            else if(raw_csv[pos] == '\r' &&  (pos < stop && raw_csv[pos+1]=='\n')){
            	stop--;
                break;
            }
			if(pos>=stop)
				break;

			pos++;
		}


		long tempPos=pos-1;

		if(dtype[col] != gdf_dtype::GDF_CATEGORY && dtype[col] != gdf_dtype::GDF_STRING){
			removePrePostWhiteSpaces2(raw_csv, &start, &tempPos);
		}


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
				case gdf_dtype::GDF_DATE32:
				{
					gdf_date32 *gdf_out = (gdf_date32 *)gdf_data[col];
					gdf_out[rec_id] = parseDateFormat(raw_csv, start, tempPos, dayfirst);
				}
					break;
				case gdf_dtype::GDF_DATE64:
				{
					gdf_date64 *gdf_out = (gdf_date64 *)gdf_data[col];
					gdf_out[rec_id] = parseDateTimeFormat(raw_csv, start, tempPos, dayfirst);
				}
					break;
				case gdf_dtype::GDF_TIMESTAMP:
				{
					int64_t *gdf_out = (int64_t *)gdf_data[col];
					gdf_out[rec_id] = convertStrtoInt<int64_t>(raw_csv, start, tempPos);
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

			// set the valid bitmap - all bits were set to 0 to start
			int bitmapIdx 	= whichBitmap(rec_id);  	// which bitmap
			int bitIdx		= whichBit(rec_id);		// which bit - over an 8-bit index
			setBit(valid[col]+bitmapIdx, bitIdx);		// This is done with atomics

			atomicAdd((unsigned long long int*)&num_valid[col],(unsigned long long int)1);
		}
		else if(dtype[col]==gdf_dtype::GDF_STRING){
			str_cols[stringCol][rec_id].first 	= NULL;
			str_cols[stringCol][rec_id].second 	= 0;
			stringCol++;
		}

		pos++;
		start=pos;
		col++;	
	}
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


