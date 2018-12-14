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
#include "datetime_parser.cuh"

#include "cudf.h"
#include "utilities/error_utils.h"
 
#include "rmm/rmm.h"

#include "NVStrings.h"

constexpr int32_t HASH_SEED = 33;

using namespace std;

//-- define the structure for raw data handling - for internal use
typedef struct raw_csv_ {
    char *				data;			// on-device: the raw unprocessed CSV data - loaded as a large char * array
    unsigned long long*	d_num_records;	// on-device: Number of records.
    unsigned long long*	recStart;		// on-device: Starting position of the records.

    char				delimiter;		// host: the delimiter
    char				terminator;		// host: the line terminator

    char				quotechar;		// host: the quote character
    bool				keepquotes;		// host: indicates to keep the start and end quotechar
    bool				doublequote;	// host: indicates to interpret two consecutive quotechar as a single

    long				num_bytes;		// host: the number of bytes in the data
    long				num_bits;		// host: the number of 64-bit bitmaps (different than valid)
	unsigned long long 	num_records;  	// host: number of records (per column)
	// int				num_cols;		// host: number of columns
	int					num_active_cols;	// host: number of columns that will be return to user.
	int					num_actual_cols;	// host: number of columns in the file --- based on the number of columns in header
    vector<gdf_dtype>	dtypes;			// host: array of dtypes (since gdf_columns are not created until end)
    vector<string>		col_names;		// host: array of column names
    bool* 				h_parseCol;		// host   : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.
    bool* 				d_parseCol;		// device : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.
    long 				header_row;		// Row id of the header
    bool				dayfirst;
    char				decimal;
    char				thousands;
} raw_csv_t;

typedef struct column_data_ {
	unsigned long long countFloat;
	unsigned long long countDateAndTime;
	unsigned long long countString;
	unsigned long long countInt8;
	unsigned long long countInt16;
	unsigned long long countInt32;
	unsigned long long countInt64;	
	unsigned long long countNULL;
} column_data_t;

typedef struct parsing_opts_ {
	char				delimiter;
	char				terminator;
	char				quotechar;
	bool				keepquotes;
	char				decimal;
	char				thousands;
} parsing_opts_t;

using string_pair = std::pair<const char*,size_t>;

//
//---------------create and process ---------------------------------------------
//
gdf_error parseArguments(csv_read_arg *args, raw_csv_t *csv);
// gdf_error getColNamesAndTypes(const char **col_names, const  char **dtypes, raw_csv_t *d);
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
gdf_error launch_dataConvertColumns(raw_csv_t * raw_csv, void** d_gdf,  gdf_valid_type** valid, gdf_dtype* d_dtypes, string_pair	**str_cols, long row_offset, unsigned long long *);

gdf_error launch_dataTypeDetection(raw_csv_t * raw_csv, long row_offset, column_data_t* d_columnData);

__global__ void countRecords(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, unsigned long long* num_records);
__global__ void storeRecordStart(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, unsigned long long* num_records,unsigned long long* recStart) ;
__global__ void convertCsvToGdf(char *csv, const parsing_opts_t opts, unsigned long long num_records, int num_columns,bool *parseCol,unsigned long long *recStart,gdf_dtype *dtype,void **gdf_data,gdf_valid_type **valid,string_pair **str_cols,unsigned long long row_offset, long header_row,bool dayfirst,unsigned long long *num_valid);
__global__ void dataTypeDetection(char *raw_csv, const parsing_opts_t opts, unsigned long long num_records, int  num_columns, bool  *parseCol, unsigned long long *recStart, unsigned long long row_offset, long header_row, column_data_t* d_columnData);

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




std::string stringType(gdf_dtype dt){

	switch (dt){
		case GDF_STRING: return std::string("str");
		case GDF_DATE64: return std::string("date64");
		case GDF_CATEGORY: return std::string("category");
		case GDF_FLOAT64: return std::string("float64");
		case GDF_INT8: return std::string("int8");
		case GDF_INT16: return std::string("int16");
		case GDF_INT32: return std::string("int32");
		case GDF_INT64: return std::string("int64");
		default:
			return "long";
	}


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
 * 		file_path			-	file location to read from	- currently the file cannot be compressed
 * 		num_cols			-	number of columns in the names and dtype arrays
 * 		names				-	ordered List of column names, this is a required field
 * 		dtype				-	ordered List of data types, this is required
 *
 * 	Optional
 * 		lineterminator		-	define the line terminator character.  Default is '\n'
 * 		delimiter			-	define the field separator, default is ','.  This argument is also called 'sep'
 *
 * 		quotechar;				define the character used to denote start and end of a quoted item
 * 		quoting;				treat string fields as quoted item and remove the first and last quotechar
 * 		nodoublequote;			do not interpret two consecutive quotechar as a single quotechar
 *
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
 *  	num_rows_out		-	Out: return the number of rows read in
 *  	gdf_column **data	-	Out: return the array of *gdf_columns
 *
 *
 * @return gdf_error
 *
 */
gdf_error read_csv(csv_read_arg *args)
{
	gdf_error error = gdf_error::GDF_SUCCESS;

	//-----------------------------------------------------------------------------
	// create the CSV data structure - this will be filled in as the CSV data is processed.
	// Done first to validate data types
	raw_csv_t * raw_csv = new raw_csv_t;
	// error = parseArguments(args, raw_csv);
	raw_csv->num_actual_cols	= args->num_cols;
	raw_csv->num_active_cols	= args->num_cols;
	raw_csv->num_records		= 0;

	if(args->delim_whitespace == true) {
		raw_csv->delimiter = ' ';
	} else {
		raw_csv->delimiter = args->delimiter;
	}

	if(args->windowslinetermination) {
		raw_csv->terminator = '\n';
	} else {
		raw_csv->terminator = args->lineterminator;
	}

	raw_csv->quotechar = args->quotechar;
	if(raw_csv->quotechar != '\0') {
		raw_csv->keepquotes = !args->quoting;
		raw_csv->doublequote = args->doublequote;
	} else {
		raw_csv->keepquotes = true;
		raw_csv->doublequote = false;
	}

	raw_csv->dayfirst = args->dayfirst;
	raw_csv->decimal = args->decimal;
	raw_csv->thousands = args->thousands == nullptr ? '\0' : *args->thousands;

	if (raw_csv->decimal == raw_csv->delimiter)
	{ 
		checkError(GDF_INVALID_API_CALL, "Decimal point cannot be the same as the delimiter");
	}
	if (raw_csv->thousands == raw_csv->delimiter)
	{ 
		checkError(GDF_INVALID_API_CALL, "Thousands separator cannot be the same as the delimiter");
	}

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

    if (map_data == MAP_FAILED || raw_csv->num_bytes==0) { close(fd); checkError(GDF_C_ERROR, "Error mapping file"); }

	//-----------------------------------------------------------------------------
	//---  create a structure to hold variables used to parse the CSV data
	error = updateRawCsv( (const char *)map_data, (long)raw_csv->num_bytes, raw_csv );
	checkError(error, "call to createRawCsv");

	//-----------------------------------------------------------------------------
	// find the record and fields points (in bitmaps)
	error = launch_countRecords(raw_csv);
	checkError(error, "call to record counter");

	//-----------------------------------------------------------------------------
	//-- Allocate space to hold the record starting point
	RMM_TRY( RMM_ALLOC((void**)&(raw_csv->recStart), (sizeof(unsigned long long) * (raw_csv->num_records + 1)), 0) ); 
	CUDA_TRY( cudaMemset(raw_csv->d_num_records,	0, 		(sizeof(unsigned long long) )) ) ;

	//-----------------------------------------------------------------------------
	//-- Scan data and set the starting positions
	error = launch_storeRecordStart(raw_csv);
	checkError(error, "call to record initial position store");

	// Previous kernel stores the record positions as encountered by all threads
	// Sort the record positions as subsequent processing may require filtering
	// certain rows or other processing on specific records
	thrust::sort(thrust::device, raw_csv->recStart, raw_csv->recStart + raw_csv->num_records + 1);

	// Currently, ignoring lineterminations within quotes is handled by recording
	// the records of both, and then filtering out the records that is a quotechar
	// or a linetermination within a quotechar pair. The future major refactoring
	// of csv_reader and its kernels will probably use a different tactic.
	if (raw_csv->quotechar != '\0') {
		const size_t recTotalSize = sizeof(unsigned long long) * (raw_csv->num_records + 1);

		unsigned long long *h_recStart = (unsigned long long*)malloc(recTotalSize);
		CUDA_TRY( cudaMemcpy(h_recStart, raw_csv->recStart, recTotalSize, cudaMemcpyDeviceToHost) );

		const char *h_data = (const char *)(map_data);
		unsigned long long recCount = raw_csv->num_records;

		bool quotation = false;
		for (size_t i = 1; i < raw_csv->num_records; ++i) {
			if (h_data[h_recStart[i] - 1] == raw_csv->quotechar) {
				quotation = !quotation;
				h_recStart[i] = raw_csv->num_bytes;
				recCount--;
			}
			else if (quotation) {
				h_recStart[i] = raw_csv->num_bytes;
				recCount--;
			}
		}

		CUDA_TRY( cudaMemcpy(raw_csv->recStart, h_recStart, recTotalSize, cudaMemcpyHostToDevice) );
		thrust::sort(thrust::device, raw_csv->recStart, raw_csv->recStart + raw_csv->num_records + 1);
		raw_csv->num_records = recCount;
		
		free(h_recStart);
	}

	//-----------------------------------------------------------------------------
	//-- Acquire header row of 

	int h_num_cols=0, h_dup_cols_removed=0;

	int skip_header=0;

	// Check if the user gave us a list of column names
	if(args->names==NULL){

		// Getting the first row of data from the file. We will parse the data to find lineterminator as
		// well as the column delimiter.
		char* cmap_data = (char *)map_data;

		unsigned long long c=0;

		raw_csv->header_row=0;
		if (args->header>=0){
			raw_csv->header_row = args->header;
		}
		if(raw_csv->header_row > (long)raw_csv->num_records){
			checkError(GDF_FILE_ERROR, "Number of records is smaller than the id of the specified header row");
		}

		unsigned long long headerPositions[2];
		CUDA_TRY( cudaMemcpy(headerPositions,raw_csv->recStart + raw_csv->header_row, sizeof(unsigned long long)*2, cudaMemcpyDeviceToHost));
		unsigned long long start = headerPositions[0];
		unsigned long long stop  = headerPositions[1];

		c=start;
		while(c<stop){
			if (cmap_data[c]==args->lineterminator){
				h_num_cols++;
				break;
			}
			else if(cmap_data[c] == '\r' && (c+1L)<(unsigned long long)raw_csv->num_bytes && cmap_data[c+1] == '\n'){
				h_num_cols++;
				break;
			}else if (cmap_data[c]==args->delimiter)
				h_num_cols++;
			c++;
		}

		unsigned long long prev=0;
		c=start;

		raw_csv->col_names.clear();

		if(args->header>=0){
			h_num_cols=0;
			// Storing the names of the columns into a vector of strings
			while(c<=stop){
				if (cmap_data[c]==args->delimiter || cmap_data[c]==args->lineterminator){
					std::string colName(cmap_data +prev,c-prev );
					prev=c+1;
					raw_csv->col_names.push_back(colName);
					h_num_cols++;
				}
				c++;
			}
			skip_header=1;
		}else{
			for (int i = 0; i<h_num_cols; i++){
				std::string newColName = std::to_string(i);
				raw_csv->col_names.push_back(newColName);
			}
		}
		// Allocating a boolean array that will use to state if a column needs to read or filtered.


		raw_csv->h_parseCol = (bool*)malloc(sizeof(bool) * (h_num_cols));
		RMM_TRY( RMM_ALLOC((void**)&raw_csv->d_parseCol,(sizeof(bool) * (h_num_cols)),0 ) );
		for (int i = 0; i<h_num_cols; i++)
			raw_csv->h_parseCol[i]=true;

		// Looking for duplicates
		for (auto it = raw_csv->col_names.begin(); it != raw_csv->col_names.end(); it++){
			bool found_dupe = false;
			for (auto it2 = (it+1); it2 != raw_csv->col_names.end(); it2++){
				if (*it==*it2){
					found_dupe=true;
					break;
				}
			}
			if(found_dupe){
				int count=1;
				for (auto it2 = (it+1); it2 != raw_csv->col_names.end(); it2++){
					if (*it==*it2){
						if(args->mangle_dupe_cols){
							// Replace all the duplicates of column X with X.1,X.2,... First appearance stays as X.
							std::string newColName  = *it2;
							newColName += "." + std::to_string(count); 
							count++;
							*it2 = newColName;							
						} else{
							// All duplicate fields will be ignored.
							int pos=std::distance(raw_csv->col_names.begin(), it2);
							raw_csv->h_parseCol[pos]=false;
							h_dup_cols_removed++;
						}
					}
				}
			}
		}

		raw_csv->num_actual_cols = h_num_cols;							// Actuaul number of columns in the CSV file
		raw_csv->num_active_cols = h_num_cols-h_dup_cols_removed;		// Number of fields that need to be processed based on duplicatation fields

		CUDA_TRY(cudaMemcpy(raw_csv->d_parseCol, raw_csv->h_parseCol, sizeof(bool) * (h_num_cols), cudaMemcpyHostToDevice));
	}
	else {
		raw_csv->h_parseCol = (bool*)malloc(sizeof(bool) * (args->num_cols));
		RMM_TRY( RMM_ALLOC((void**)&raw_csv->d_parseCol,(sizeof(bool) * (args->num_cols)),0 ) );

		for (int i = 0; i<raw_csv->num_actual_cols; i++){
			raw_csv->h_parseCol[i]=true;
			std::string col_name 	= args->names[i];
			raw_csv->col_names.push_back(col_name);

		}
		CUDA_TRY(cudaMemcpy(raw_csv->d_parseCol, raw_csv->h_parseCol, sizeof(bool) * (args->num_cols), cudaMemcpyHostToDevice));
	}

	// User can give
	if (args->use_cols_int!=NULL || args->use_cols_char!=NULL){
		if(args->use_cols_int!=NULL){
			for (int i = 0; i<raw_csv->num_actual_cols; i++)
				raw_csv->h_parseCol[i]=false;
			for(int i=0; i < args->use_cols_int_len; i++){
				int pos = args->use_cols_int[i];
				raw_csv->h_parseCol[pos]=true;
			}
			raw_csv->num_active_cols = args->use_cols_int_len;
		}else{
			for (int i = 0; i<raw_csv->num_actual_cols; i++)
				raw_csv->h_parseCol[i]=false;
			int countFound=0;
			for(int i=0; i < args->use_cols_char_len; i++){
				std::string colName(args->use_cols_char[i]);
				for (auto it = raw_csv->col_names.begin(); it != raw_csv->col_names.end(); it++){
					if(colName==*it){
						countFound++;
						int pos=std::distance(raw_csv->col_names.begin(), it);
						raw_csv->h_parseCol[pos]=true;
						break;
					}
				}
			}
			raw_csv->num_active_cols = countFound;
		}
		CUDA_TRY(cudaMemcpy(raw_csv->d_parseCol, raw_csv->h_parseCol, sizeof(bool) * (raw_csv->num_actual_cols), cudaMemcpyHostToDevice));
	}

	raw_csv->num_records -= (args->skiprows + args->skipfooter); 
	if(skip_header==0){
		raw_csv->header_row=-1;
	}else{
		raw_csv->num_records-=1;
	}

	//-----------------------------------------------------------------------------
	//---  done with host data
	close(fd);
	munmap(map_data, raw_csv->num_bytes);


	//-----------------------------------------------------------------------------
	//--- Auto detect types of the vectors

	// if(args->dtype==NULL){
	if(args->names==NULL){

		column_data_t *d_ColumnData,*h_ColumnData;

		h_ColumnData = (column_data_t*)malloc(sizeof(column_data_t) * (raw_csv->num_active_cols));
		RMM_TRY( RMM_ALLOC((void**)&d_ColumnData,(sizeof(column_data_t) * (raw_csv->num_active_cols)),0 ) );

		CUDA_TRY( cudaMemset(d_ColumnData,	0, 	(sizeof(column_data_t) * (raw_csv->num_active_cols)) ) ) ;

		launch_dataTypeDetection(raw_csv, args->skiprows, d_ColumnData);

		CUDA_TRY( cudaMemcpy(h_ColumnData,d_ColumnData, sizeof(column_data_t) * (raw_csv->num_active_cols), cudaMemcpyDeviceToHost));

	    vector<gdf_dtype>	d_detectedTypes;			// host: array of dtypes (since gdf_columns are not created until end)

		raw_csv->dtypes.clear();

		for(int col = 0; col < raw_csv->num_active_cols; col++){
			unsigned long long countInt = h_ColumnData[col].countInt8+h_ColumnData[col].countInt16+
										  h_ColumnData[col].countInt32+h_ColumnData[col].countInt64;

			if (h_ColumnData[col].countNULL == raw_csv->num_records){
				d_detectedTypes.push_back(GDF_INT8); // Entire column is NULL. Allocating the smallest amount of memory
			} else if(h_ColumnData[col].countString>0L){
				d_detectedTypes.push_back(GDF_CATEGORY); // For auto-detection, we are currently not supporting strings.
			} else if(h_ColumnData[col].countDateAndTime>0L){
				d_detectedTypes.push_back(GDF_DATE64);
			} else if(h_ColumnData[col].countFloat > 0L  ||  
				(h_ColumnData[col].countFloat==0L && countInt >0L && h_ColumnData[col].countNULL >0L) ) {
				// The second condition has been added to conform to PANDAS which states that a colum of 
				// integers with a single NULL record need to be treated as floats.
				d_detectedTypes.push_back(GDF_FLOAT64);
			}
			else { 
				d_detectedTypes.push_back(GDF_INT64);
			}
		}

		raw_csv->dtypes=d_detectedTypes;

		free(h_ColumnData);
		RMM_TRY( RMM_FREE( d_ColumnData, 0 ) );
	}
	else{
		for ( int x = 0; x < raw_csv->num_actual_cols; x++) {

			std::string temp_type 	= args->dtype[x];
			gdf_dtype col_dtype		= convertStringToDtype( temp_type );

			if (col_dtype == GDF_invalid)
				return GDF_UNSUPPORTED_DTYPE;

			raw_csv->dtypes.push_back(col_dtype);
		}
	}


	//-----------------------------------------------------------------------------
	//--- allocate space for the results
	gdf_column **cols = (gdf_column **)malloc( sizeof(gdf_column *) * raw_csv->num_active_cols);

	void **d_data,**h_data;
	gdf_valid_type **d_valid,**h_valid;
    unsigned long long	*d_valid_count,*h_valid_count;
	gdf_dtype *d_dtypes,*h_dtypes;





	h_dtypes 		= (gdf_dtype*)malloc (	sizeof(gdf_dtype)* (raw_csv->num_active_cols));
	h_valid_count	= (unsigned long long*)malloc (	sizeof(unsigned long long)* (raw_csv->num_active_cols));
	h_data 			= (void**)malloc (	sizeof(void*)* (raw_csv->num_active_cols));
	h_valid 		= (gdf_valid_type**)malloc (	sizeof(gdf_valid_type*)* (raw_csv->num_active_cols));

	RMM_TRY( RMM_ALLOC((void**)&d_dtypes, 		(sizeof(gdf_dtype) 			* raw_csv->num_active_cols), 0 ) );
	RMM_TRY( RMM_ALLOC((void**)&d_data, 		(sizeof(void *)				* raw_csv->num_active_cols), 0 ) );
	RMM_TRY( RMM_ALLOC((void**)&d_valid, 		(sizeof(gdf_valid_type *)	* raw_csv->num_active_cols), 0 ) );
	RMM_TRY( RMM_ALLOC((void**)&d_valid_count, 	(sizeof(unsigned long long) * raw_csv->num_active_cols), 0 ) );
	CUDA_TRY( cudaMemset(d_valid_count,	0, 		(sizeof(unsigned long long)	* raw_csv->num_active_cols)) );


	int stringColCount=0;
	for (int col = 0; col < raw_csv->num_active_cols; col++) {
		if(raw_csv->dtypes[col]==gdf_dtype::GDF_STRING)
			stringColCount++;
	}

	string_pair **h_str_cols = NULL, **d_str_cols = NULL;

	if (stringColCount > 0 ) {
		h_str_cols = (string_pair**) malloc ((sizeof(string_pair *)	* stringColCount));
		RMM_TRY( RMM_ALLOC((void**)&d_str_cols, 	(sizeof(string_pair *)		* stringColCount), 0) );

		for (int col = 0; col < stringColCount; col++) {
			RMM_TRY( RMM_ALLOC((void**)(h_str_cols + col), sizeof(string_pair) * (raw_csv->num_records), 0) );
		}

		CUDA_TRY(cudaMemcpy(d_str_cols, h_str_cols, sizeof(string_pair *)	* stringColCount, cudaMemcpyHostToDevice));
	}

	for (int col = 0; col < raw_csv->num_active_cols; col++) {

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
		h_dtypes[col] 	= raw_csv->dtypes[col];
		h_data[col] 	= gdf->data;
		h_valid[col] 	= gdf->valid;
	}
	CUDA_TRY( cudaMemcpy(d_dtypes,h_dtypes, sizeof(gdf_dtype) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));
	CUDA_TRY( cudaMemcpy(d_data,h_data, sizeof(void*) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));
	CUDA_TRY( cudaMemcpy(d_valid,h_valid, sizeof(gdf_valid_type*) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));

	free(h_dtypes); 
	free(h_valid); 
	free(h_data); 
	
	launch_dataConvertColumns(raw_csv,d_data, d_valid, d_dtypes,d_str_cols, args->skiprows, d_valid_count);
	cudaDeviceSynchronize();

	stringColCount=0;
	for (int col = 0; col < raw_csv->num_active_cols; col++) {

		gdf_column *gdf = cols[col];

		if (gdf->dtype != gdf_dtype::GDF_STRING)
			continue;

		NVStrings* const stringCol = NVStrings::create_from_index(h_str_cols[stringColCount],size_t(raw_csv->num_records));
		if ((raw_csv->quotechar != '\0') && (raw_csv->doublequote==true)) {
			// In PANDAS, default of enabling doublequote for two consecutive
			// quotechar in quote fields results in reduction to single
			std::string quotechar = std::string(&raw_csv->quotechar);
			std::string doublequotechar = quotechar + raw_csv->quotechar;
			gdf->data = stringCol->replace(doublequotechar.c_str(), quotechar.c_str());
			NVStrings::destroy(stringCol);
		}
		else {
			gdf->data = stringCol;
		}

		RMM_TRY( RMM_FREE( h_str_cols [stringColCount], 0 ) );

		stringColCount++;
	}


	CUDA_TRY( cudaMemcpy(h_valid_count,d_valid_count, sizeof(unsigned long long) * (raw_csv->num_active_cols), cudaMemcpyDeviceToHost));

	//--- set the null count
	for ( int col = 0; col < raw_csv->num_active_cols; col++) {
		cols[col]->null_count = raw_csv->num_records - h_valid_count[col];
	}

	free(h_valid_count); 

	// free up space that is no longer needed
	if (h_str_cols != NULL)
		free ( h_str_cols);

	free(raw_csv->h_parseCol);

	if (d_str_cols != NULL)
		RMM_TRY( RMM_FREE( d_str_cols, 0 ) ); 

	RMM_TRY( RMM_FREE( d_valid, 0 ) );
	RMM_TRY( RMM_FREE( d_valid_count, 0 ) );
	RMM_TRY( RMM_FREE( d_dtypes, 0 ) );
	RMM_TRY( RMM_FREE( d_data, 0 ) ); 

	RMM_TRY( RMM_FREE( raw_csv->recStart, 0 ) ); 
	RMM_TRY( RMM_FREE( raw_csv->d_parseCol, 0 ) ); 
	RMM_TRY( RMM_FREE( raw_csv->d_num_records, 0 ) ); 
	CUDA_TRY( cudaFree ( raw_csv->data) );


	args->data 			= cols;
	args->num_cols_out	= raw_csv->num_active_cols;
	args->num_rows_out	= raw_csv->num_records;

	delete raw_csv;
	return error;
}



/*
 * What is passed in is the data type as a string, need to convert that into gdf_dtype enum
 */
gdf_dtype convertStringToDtype(std::string &dtype) {

	if (dtype.compare( "str") == 0) 		return GDF_STRING;
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

	CUDA_TRY( cudaMallocManaged ((void**)&raw->data, 		(sizeof(char)		* num_bytes)));
	// RMM_TRY( RMM_ALLOC((void**)&raw->data, 		(sizeof(char)		* num_bytes),0 ));

	RMM_TRY( RMM_ALLOC((void**)&raw->d_num_records, sizeof(unsigned long long),0) );

	CUDA_TRY( cudaMemcpy(raw->data, data, num_bytes, cudaMemcpyHostToDevice));
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
	RMM_TRY( RMM_ALLOC((void**)&gdf->valid, (sizeof(gdf_valid_type) * num_bitmaps), 0) );
	CUDA_TRY(cudaMemset(gdf->valid, 0, (sizeof(gdf_valid_type) 	* num_bitmaps)) );

	int elementSize=0;
	//--- Allocate space for the data
	switch(gdf->dtype) {
		case gdf_dtype::GDF_INT8:
			elementSize = sizeof(int8_t);
			break;
		case gdf_dtype::GDF_INT16:
			elementSize = sizeof(int16_t);
			break;
		case gdf_dtype::GDF_INT32:
			elementSize = sizeof(int32_t);
			break;
		case gdf_dtype::GDF_INT64:
			elementSize = sizeof(int64_t);
			break;
		case gdf_dtype::GDF_FLOAT32:
			elementSize = sizeof(float);
			break;
		case gdf_dtype::GDF_FLOAT64:
			elementSize = sizeof(double);
			break;
		case gdf_dtype::GDF_DATE32:
			elementSize = sizeof(gdf_date32);
			break;
		case gdf_dtype::GDF_DATE64:
			elementSize = sizeof(gdf_date64);
			break;
		case gdf_dtype::GDF_TIMESTAMP:
			elementSize = sizeof(int64_t);
			break;
		case gdf_dtype::GDF_CATEGORY:
			elementSize = sizeof(gdf_category);
			break;
		case gdf_dtype::GDF_STRING:
			return gdf_error::GDF_SUCCESS;
			// Memory for gdf->data allocated by string class eventually
		default:
			return GDF_UNSUPPORTED_DTYPE;
	}
	
	RMM_TRY( RMM_ALLOC((void**)&gdf->data, elementSize * N, 0) );

	return gdf_error::GDF_SUCCESS;
}


//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------


gdf_error launch_countRecords(raw_csv_t * csvData) {

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, countRecords) );

	// Calculate actual block count to use based on bitmap count
	// Each bitmap is for a 64-byte chunk, and each data index is bitmap ID * 64
	int gridSize = (csvData->num_bits + blockSize - 1) / blockSize;

	countRecords <<< gridSize, blockSize >>> (
		csvData->data, csvData->terminator, csvData->quotechar,
		csvData->num_bytes, csvData->num_bits, csvData->d_num_records
	);

	CUDA_TRY(cudaGetLastError());

	long recs=-1;
	CUDA_TRY(cudaMemcpy(&recs, csvData->d_num_records, sizeof(long), cudaMemcpyDeviceToHost));
	csvData->num_records=recs;

	CUDA_TRY(cudaGetLastError());

	return GDF_SUCCESS;
}


__global__ void countRecords(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, unsigned long long* num_records) {

	// thread IDs range per block, so also need the block id
	long tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid >= num_bits)
		return;

	// data ID is a multiple of 64
	long did = tid * 64L;

	char *raw = (data + did);

	long byteToProcess = ((did + 64L) < num_bytes) ? 64L : (num_bytes - did);

	// process the data
	long tokenCount = 0;
	for (long x = 0; x < byteToProcess; x++) {
		
		// Scan and log records. If quotations are enabled, then also log quotes
		// for a postprocess ignore, as the chunk here has limited visibility.
		if ((raw[x] == terminator) || (quotechar != '\0' && raw[x] == quotechar)) {
			tokenCount++;
		} else if (raw[x] == '\r' && (x+1L)<num_bytes && raw[x +1] == '\n') {
			x++;
			tokenCount++;
		}

	}
	atomicAdd((unsigned long long int*)num_records,(unsigned long long int)tokenCount);
}


gdf_error launch_storeRecordStart(raw_csv_t * csvData) {

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, storeRecordStart) );

	// Calculate actual block count to use based on bitmap count
	// Each bitmap is for a 64-byte chunk, and each data index is bitmap ID * 64
	int gridSize = (csvData->num_bits + blockSize - 1) / blockSize;

	storeRecordStart <<< gridSize, blockSize >>> (
		csvData->data, csvData->terminator, csvData->quotechar,
		csvData->num_bytes, csvData->num_bits, csvData->d_num_records,
		csvData->recStart
	);

	CUDA_TRY( cudaGetLastError() );
	return GDF_SUCCESS;
}


__global__ void storeRecordStart(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, unsigned long long* num_records,unsigned long long* recStart) {

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
	for (long x = 0; x < byteToProcess; x++) {

		// Scan and log records. If quotations are enabled, then also log quotes
		// for a postprocess ignore, as the chunk here has limited visibility.
		if ((raw[x] == terminator) || (quotechar != '\0' && raw[x] == quotechar)) {

			long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
			recStart[pos]=did+x+1;

		} else if (raw[x] == '\r' && (x+1L)<num_bytes && raw[x +1] == '\n') {

			x++;
			long pos = atomicAdd((unsigned long long int*)num_records,(unsigned long long int)1);
			recStart[pos]=did+x+1;
		}

	}
}


//----------------------------------------------------------------------------------------------------------------


gdf_error launch_dataConvertColumns(raw_csv_t *raw_csv, void **gdf, gdf_valid_type** valid, gdf_dtype* d_dtypes,string_pair **str_cols, long row_offset, unsigned long long *num_valid) {

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, convertCsvToGdf) );

	// Calculate actual block count to use based on records count
	int gridSize = (raw_csv->num_records + blockSize - 1) / blockSize;

	parsing_opts_t opts;
	opts.delimiter		= raw_csv->delimiter;
	opts.terminator		= raw_csv->terminator;
	opts.quotechar		= raw_csv->quotechar;
	opts.keepquotes		= raw_csv->keepquotes;
	opts.decimal		= raw_csv->decimal;
	opts.thousands		= raw_csv->thousands;

	convertCsvToGdf <<< gridSize, blockSize >>>(
		raw_csv->data,
		opts,
		raw_csv->num_records,
		raw_csv->num_actual_cols,
		raw_csv->d_parseCol,
		raw_csv->recStart,
		d_dtypes,
		gdf,
		valid,
		str_cols,
		row_offset,
		raw_csv->header_row,
		raw_csv->dayfirst,
		num_valid
	);

	CUDA_TRY( cudaGetLastError() );
	return GDF_SUCCESS;
}


/*
 * Data is processed in one row\record at a time - so the number of total threads (tid) is equal to the number of rows.
 *
 */
__global__ void convertCsvToGdf(
		char 			*raw_csv,
		const parsing_opts_t	 	opts,
		unsigned long long  num_records,
		int  			num_columns,
		bool  			*parseCol,
		unsigned long long 			*recStart,
		gdf_dtype 		*dtype,
		void			**gdf_data,
		gdf_valid_type 	**valid,
		string_pair		**str_cols,
		unsigned long long 			row_offset,
		long 			header_row,
		bool			dayfirst,
		unsigned long long			*num_valid
		)
{
	// thread IDs range per block, so also need the block id
	long	rec_id  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( rec_id >= num_records)
		return;

	long extraOff=0;
	if(rec_id>=header_row && header_row>=0)
		extraOff=1;

	long start 		= recStart[rec_id + row_offset + extraOff];
	long stop 		= recStart[rec_id + 1 + row_offset + extraOff];

	long pos 		= start;
	int  col 		= 0;
	int  actual_col = 0;
	int  stringCol 	= 0;
	bool quotation	= false;

	while(col<num_columns){

		if(start>stop)
			break;

		while(true){
			// Use simple logic to ignore control chars between any quote seq
			// Handles nominal cases including doublequotes within quotes, but
			// may not output exact failures as PANDAS for malformed fields
			if(raw_csv[pos] == opts.quotechar){
				quotation = !quotation;
			}
			else if(quotation==false){
				if(raw_csv[pos] == opts.delimiter){
					break;
				}
				else if(raw_csv[pos] == opts.terminator){
					break;
				}
				else if(raw_csv[pos] == '\r' && ((pos+1) < stop && raw_csv[pos+1] == '\n')){
					stop--;
					break;
				}
			}
			if(pos>=stop)
				break;
			pos++;
		}

		if(parseCol[col]==true){

			long tempPos=pos-1;

			if(dtype[col] != gdf_dtype::GDF_CATEGORY && dtype[col] != gdf_dtype::GDF_STRING){
				removePrePostWhiteSpaces2(raw_csv, &start, &tempPos);
			}


			if(start<=(tempPos)) { // Empty strings are not legal values

				switch(dtype[col]) {
					case gdf_dtype::GDF_INT8:
					{
						int8_t *gdf_out = (int8_t *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoInt<int8_t>(raw_csv, start, tempPos, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_INT16: {
						int16_t *gdf_out = (int16_t *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoInt<int16_t>(raw_csv, start, tempPos, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_INT32:
					{
						int32_t *gdf_out = (int32_t *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoInt<int32_t>(raw_csv, start, tempPos, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_INT64:
					{
						int64_t *gdf_out = (int64_t *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoInt<int64_t>(raw_csv, start, tempPos, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_FLOAT32:
					{
						float *gdf_out = (float *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoFloat<float>(raw_csv, start, tempPos, opts.decimal, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_FLOAT64:
					{
						double *gdf_out = (double *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoFloat<double>(raw_csv, start, tempPos, opts.decimal, opts.thousands);
					}
						break;
					case gdf_dtype::GDF_DATE32:
					{
						gdf_date32 *gdf_out = (gdf_date32 *)gdf_data[actual_col];
						gdf_out[rec_id] = parseDateFormat(raw_csv, start, tempPos, dayfirst);
					}
						break;
					case gdf_dtype::GDF_DATE64:
					{
						gdf_date64 *gdf_out = (gdf_date64 *)gdf_data[actual_col];
						gdf_out[rec_id] = parseDateTimeFormat(raw_csv, start, tempPos, dayfirst);
					}
						break;
					case gdf_dtype::GDF_TIMESTAMP:
					{
						int64_t *gdf_out = (int64_t *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoInt<int64_t>(raw_csv, start, tempPos, opts.thousands);
					}
					break;
					case gdf_dtype::GDF_CATEGORY:
					{
						gdf_category *gdf_out = (gdf_category *)gdf_data[actual_col];
						gdf_out[rec_id] = convertStrtoHash(raw_csv, start, pos, HASH_SEED);
					}
						break;
					case gdf_dtype::GDF_STRING:
					{
						long end = pos;
						if(opts.keepquotes==false){
							if((raw_csv[start] == opts.quotechar) && (raw_csv[end-1] == opts.quotechar)){
								start++;
								end--;
							}
						}
						str_cols[stringCol][rec_id].first	= raw_csv+start;
						str_cols[stringCol][rec_id].second	= size_t(end-start);
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
			actual_col++;
		}
		pos++;
		start=pos;
		col++;

	}
}



//----------------------------------------------------------------------------------------------------------------


gdf_error launch_dataTypeDetection(
	raw_csv_t * raw_csv, 
	long row_offset,
	column_data_t* d_columnData) 
{
	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dataTypeDetection) );

	// Calculate actual block count to use based on records count
	int gridSize = (raw_csv->num_records + blockSize - 1) / blockSize;

	parsing_opts_t opts;
	opts.delimiter		= raw_csv->delimiter;
	opts.terminator		= raw_csv->terminator;
	opts.quotechar		= raw_csv->quotechar;
	opts.keepquotes		= raw_csv->keepquotes;

	dataTypeDetection <<< gridSize, blockSize >>>(
		raw_csv->data,
		opts,
		raw_csv->num_records,
		raw_csv->num_actual_cols,
		raw_csv->d_parseCol,
		raw_csv->recStart,
		row_offset,
		raw_csv->header_row,
		d_columnData
	);

	CUDA_TRY( cudaGetLastError() );
	return GDF_SUCCESS;
}

/*
 */
__global__ void dataTypeDetection(
		char 			*raw_csv,
		const parsing_opts_t			opts,
		unsigned long long  			num_records,
		int  			num_columns,
		bool  			*parseCol,
		unsigned long long 			*recStart,
		unsigned long long  			row_offset,
		long 			header_row,
		column_data_t* d_columnData
		)
{

	// thread IDs range per block, so also need the block id
	long	rec_id  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( rec_id >= num_records)
		return;

	long extraOff=0;
	if(rec_id>=header_row && header_row>=0)
		extraOff=1;

	long start 		= recStart[rec_id + row_offset + extraOff];
	long stop 		= recStart[rec_id + 1 + row_offset + extraOff];

	long pos 		= start;
	int  col 		= 0;
	int  actual_col = 0;
	bool quotation	= false;

	// Going through all the columns of a given record
	while(col<num_columns){

		if(start>stop)
			break;

		// Finding the breaking point for each column
		while(true){
			// Use simple logic to ignore control chars between any quote seq
			// Handles nominal cases including doublequotes within quotes, but
			// may not output exact failures as PANDAS for malformed fields
			if(raw_csv[pos] == opts.quotechar){
				quotation = !quotation;
			}
			else if(quotation==false){
				if(raw_csv[pos] == opts.delimiter){
					break;
				}
				else if(raw_csv[pos] == opts.terminator){
					break;
				}
				else if(raw_csv[pos] == '\r' && ((pos+1) < stop && raw_csv[pos+1] == '\n')){
					stop--;
					break;
				}
			}
			if(pos>=stop)
				break;
			pos++;
		}


		// Checking if this is a column that the user wants --- user can filter columns
		if(parseCol[col]==true){

			long tempPos=pos-1;

			// Checking if the record is NULL
			if(start>(tempPos)){
				atomicAdd(& d_columnData[actual_col].countNULL, 1L);
				pos++;
				start=pos;
				col++;
				actual_col++;
				continue;	
			}

			long countNumber=0;
			long countDecimal=0;
			long countSlash=0;
			long countDash=0;
			long countColon=0;
			long countString=0;

			long strLen=pos-start;

			// Remove all pre and post white-spaces.  We might find additional NULL fields if the entire entry is made up of only spaces.
			removePrePostWhiteSpaces2(raw_csv, &start, &tempPos);
			for(long startPos=start; startPos<=tempPos; startPos++){
				if(raw_csv[startPos]>= '0' && raw_csv[startPos] <= '9'){
					countNumber++;
					continue;
				}
				// Looking for unique characters that will help identify column types.
				switch (raw_csv[startPos]){
					case '.':
						countDecimal++;break;
					case '-':
						countDash++; break;
					case '/':
						countSlash++;break;
					case ':':
						countColon++;break;
					default:
						countString++;
						break;	
				}
			}

			if(strLen==0) // Removed spaces ' ' in the pre-processing and thus we can have an empty string.
				atomicAdd(& d_columnData[actual_col].countNULL, 1L);
			// Integers have to have the length of the string or can be off by one if they start with a minus sign
			else if(countNumber==(strLen) || ( strLen>1 && countNumber==(strLen-1) && raw_csv[start]=='-') ){
				// Checking to see if we the integer value requires 8,16,32,64 bits.
				// This will allow us to allocate the exact amount of memory.
				int64_t i = convertStrtoInt<int64_t>(raw_csv, start, tempPos, opts.thousands);
				if(i >= (1L<<31)){
					atomicAdd(& d_columnData[actual_col].countInt64, 1L);
				}
				else if(i >= (1L<<15)){
					atomicAdd(& d_columnData[actual_col].countInt32, 1L);
				}
				else if(i >= (1L<<7)){
					atomicAdd(& d_columnData[actual_col].countInt16, 1L);
				}
				else
					atomicAdd(& d_columnData[actual_col].countInt8, 1L);
			}
			// Floating point numbers are made up of numerical strings, have to have a decimal sign, and can have a minus sign.
			else if((countNumber==(strLen-1) && countDecimal==1) || (strLen>2 && countNumber==(strLen-2) && raw_csv[start]=='-')){
					atomicAdd(& d_columnData[actual_col].countFloat, 1L);
			}
			// The date-time field cannot have more than 3 strings. As such if an entry has more than 3 string characters, it is not 
			// a data-time field. Also, if a string has multiple decimals, then is not a legit number.
			else if(countString > 3 || countDecimal > 1){
				atomicAdd(& d_columnData[actual_col].countString, 1L);
			}
			else {
				// A date field can have either one or two '-' or '\'. A legal combination will only have one of them.
				// To simplify the process of auto column detection, we are not covering all the date-time formation permutations.
				if((countDash>0 && countDash<=2 && countSlash==0)|| (countDash==0 && countSlash>0 && 	countSlash<=2) ){
					if((countColon<=2)){
						atomicAdd(& d_columnData[actual_col].countDateAndTime, 1L);
					}
					else{
						atomicAdd(& d_columnData[actual_col].countString, 1L);					
					}
				}
				// Default field is string type.
				else{
					atomicAdd(& d_columnData[actual_col].countString, 1L);					
				}
			}
			actual_col++;
		}
		pos++;
		start=pos;
		col++;	

	}
}

//----------------------------------------------------------------------------------------------------------------

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


