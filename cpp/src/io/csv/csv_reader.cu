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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
#include "utilities/trie.cuh"
#include "utilities/type_dispatcher.hpp"

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "io/comp/io_uncomp.h"

constexpr size_t max_chunk_bytes = 64*1024*1024; // 64MB

using std::vector;
using std::string;

using cu_reccnt_t = unsigned long long int;
using cu_recstart_t = unsigned long long int;


/**---------------------------------------------------------------------------*
 * @brief Struct used for internal parsing state
 *---------------------------------------------------------------------------**/
typedef struct raw_csv_ {
    char *				data;			// on-device: the raw unprocessed CSV data - loaded as a large char * array
    cu_recstart_t*		recStart;		// on-device: Starting position of the records.

    ParseOptions        opts;			// host: options to control parsing behavior

    long				num_bytes;		// host: the number of bytes in the data
    long				num_bits;		// host: the number of 64-bit bitmaps (different than valid)
	gdf_size_type 		num_records;  	// host: number of records loaded into device memory, and then number of records to read
	// int				num_cols;		// host: number of columns
	int					num_active_cols;	// host: number of columns that will be return to user.
	int					num_actual_cols;	// host: number of columns in the file --- based on the number of columns in header
    vector<gdf_dtype>	dtypes;			// host: array of dtypes (since gdf_columns are not created until end)
    vector<string>		col_names;		// host: array of column names
    bool* 				h_parseCol;		// host   : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.
    bool* 				d_parseCol;		// device : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.

    long        byte_range_offset;  // offset into the data to start parsing
    long        byte_range_size;    // length of the data of interest to parse

    gdf_size_type header_row;       ///< host: Row index of the header
    gdf_size_type nrows;            ///< host: Number of rows to read. -1 for all rows
    gdf_size_type skiprows;         ///< host: Number of rows to skip from the start
    gdf_size_type skipfooter;       ///< host: Number of rows to skip from the end
    std::vector<char> header;       ///< host: Header row data, for parsing column names
    string prefix;                  ///< host: Prepended to column ID if there is no header or input column names

    rmm::device_vector<int32_t>	d_trueValues;		// device: array of values to recognize as true
    rmm::device_vector<int32_t>	d_falseValues;		// device: array of values to recognize as false
    rmm::device_vector<SerialTrieNode>	d_naTrie;	// device: serialized trie of NA values
} raw_csv_t;

typedef struct column_data_ {
	unsigned long long countFloat;
	unsigned long long countDateAndTime;
	unsigned long long countString;
	unsigned long long countInt8;
	unsigned long long countInt16;
	unsigned long long countInt32;
	unsigned long long countInt64;
	gdf_size_type countNULL;
} column_data_t;

using string_pair = std::pair<const char*,size_t>;

//
//---------------create and process ---------------------------------------------
//
gdf_error parseArguments(csv_read_arg *args, raw_csv_t *csv);
// gdf_error getColNamesAndTypes(const char **col_names, const  char **dtypes, raw_csv_t *d);
gdf_error inferCompressionType(const char* compression_arg, const char* filepath, string& compression_type);
gdf_error getUncompressedHostData(const char* h_data, size_t num_bytes, 
	const string& compression, 
	vector<char>& h_uncomp_data);
gdf_error uploadDataToDevice(const char* h_uncomp_data, size_t h_uncomp_size, raw_csv_t * raw_csv);
gdf_error allocateGdfDataSpace(gdf_column *);
gdf_dtype convertStringToDtype(std::string &dtype);

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { std::cerr << "ERROR:  " << error <<  "  in "  << txt << std::endl;  return error; }

//
//---------------CUDA Kernel ---------------------------------------------
//

__device__ int findSetBit(int tid, long num_bits, uint64_t *f_bits, int x);

gdf_error launch_countRecords(const char* h_data, size_t h_size, raw_csv_t * raw_csv, gdf_size_type& rec_cnt);
gdf_error launch_storeRecordStart(const char* h_data, size_t h_size, raw_csv_t * csvData);
gdf_error launch_dataConvertColumns(raw_csv_t * raw_csv, void** d_gdf,  gdf_valid_type** valid, gdf_dtype* d_dtypes, string_pair **str_cols, unsigned long long *);

gdf_error launch_dataTypeDetection(raw_csv_t * raw_csv, column_data_t* d_columnData);

__global__ void countRecords(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, cu_reccnt_t* num_records);
__global__ void storeRecordStart(char *data, size_t chunk_offset, 
	const char terminator, const char quotechar, bool include_first_row,
	long num_bytes, long num_bits, cu_reccnt_t* num_records,
	cu_recstart_t* recStart);
__global__ void convertCsvToGdf(char *csv, const ParseOptions opts,
	gdf_size_type num_records, int num_columns, bool *parseCol,
	cu_recstart_t *recStart, gdf_dtype *dtype, SerialTrieNode *na_trie, void **gdf_data, gdf_valid_type **valid,
	string_pair **str_cols, unsigned long long *num_valid);
__global__ void dataTypeDetection(char *raw_csv, const ParseOptions opts,
	gdf_size_type num_records, int num_columns, bool *parseCol,
	cu_recstart_t *recStart, column_data_t* d_columnData);

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


/**---------------------------------------------------------------------------*
 * @brief Estimates the maximum expected length or a row, based on the number 
 * of columns
 * 
 * If the number of columns is not available, it will return a value large 
 * enough for most use cases
 * 
 * @param[in] num_columns Number of columns in the CSV file (optional)
 * 
 * @return Estimated maximum size of a row, in bytes
 *---------------------------------------------------------------------------**/
 constexpr size_t calculateMaxRowSize(int num_columns=0) noexcept {
	constexpr size_t max_row_bytes = 16*1024; // 16KB
	constexpr size_t column_bytes = 64;
	constexpr size_t base_padding = 1024; // 1KB
	if (num_columns == 0){
		// Use flat size if the number of columns is not known
		return max_row_bytes;
	}
	else {
		// Expand the size based on the number of columns, if available
		return base_padding + num_columns * column_bytes; 
	}
}
/**
* @brief Removes the first and Last quote in the string
*/
string removeQuotes(string str, char quotechar) {
	// Exclude first and last quotation char
	const size_t first_quote = str.find(quotechar);
	if (first_quote != string::npos) {
		str.erase(first_quote, 1);
	}
	const size_t  last_quote = str.rfind(quotechar);
	if (last_quote != string::npos) {
		str.erase(last_quote, 1);
	}

	return str;
}

/**
 * @brief Parse the first row to set the column names in the raw_csv parameter 
 *
 * The first row can be either the header row, or the first data row
 *
 * @param[in,out] raw_csv Structure containing the csv parsing parameters
 * and intermediate results
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
*/
gdf_error setColumnNamesFromCsv(raw_csv_t* raw_csv) {
	vector<char> first_row = raw_csv->header;
	// No header, read the first data row
	if (first_row.empty()) {
		cu_recstart_t first_row_len{};
		// If file only contains one row, raw_csv->recStart[1] is not valid
		if (raw_csv->num_records > 1) {
			CUDA_TRY(cudaMemcpy(&first_row_len, raw_csv->recStart + 1, sizeof(cu_recstart_t), cudaMemcpyDefault));
		}
		else {
			// File has one row - use the file size for the row size
			first_row_len = raw_csv->num_bytes / sizeof(char);
		}
		first_row.resize(first_row_len);
		CUDA_TRY(cudaMemcpy(first_row.data(), raw_csv->data, raw_csv->num_bytes, cudaMemcpyDefault));
	}

	int num_cols = 0;

	bool quotation	= false;
	for (size_t pos = 0, prev = 0; pos < first_row.size(); ++pos) {
		// Flip the quotation flag if current character is a quotechar
		if(first_row[pos] == raw_csv->opts.quotechar) {
			quotation = !quotation;
		}
		else if (!quotation &&
				 (first_row[pos] == raw_csv->opts.delimiter ||
				 first_row[pos] == raw_csv->opts.terminator)) {
			// Got to the end of a column
			if (raw_csv->header_row >= 0) {
				// first_row is the header, add the column name
				string new_col_name(first_row.data() + prev, pos - prev);
				raw_csv->col_names.push_back(removeQuotes(new_col_name, raw_csv->opts.quotechar));
			}
			else {
				// first_row is the first data row, add the automatically generated name
				raw_csv->col_names.push_back(raw_csv->prefix + std::to_string(num_cols));
			}
			num_cols++;

			// Skip adjacent delimiters if delim_whitespace is set
			while (raw_csv->opts.multi_delimiter &&
				   pos < first_row.size() &&
				   first_row[pos] == raw_csv->opts.delimiter && 
				   first_row[pos + 1] == raw_csv->opts.delimiter) {
				++pos;
			}
			prev = pos + 1;
		}
	}
	return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Read in a CSV file, extract all fields and return 
 * a GDF (array of gdf_columns)
 *
 * @param[in,out] args Structure containing both the the input arguments 
 * and the returned data
 *
 * @return gdf_error
 *---------------------------------------------------------------------------**/
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

	raw_csv->header_row = args->header;
	raw_csv->skiprows = args->skiprows;
	raw_csv->skipfooter = args->skipfooter;
	raw_csv->nrows = args->nrows;
	raw_csv->prefix = args->prefix == nullptr ? "" : string(args->prefix);

	if (args->delim_whitespace) {
		raw_csv->opts.delimiter = ' ';
		raw_csv->opts.multi_delimiter = true;
	} else {
		raw_csv->opts.delimiter = args->delimiter;
		raw_csv->opts.multi_delimiter = false;
	}
	if (args->windowslinetermination) {
		raw_csv->opts.terminator = '\n';
	} else {
		raw_csv->opts.terminator = args->lineterminator;
	}
	if (args->quotechar != '\0') {
		raw_csv->opts.quotechar = args->quotechar;
		raw_csv->opts.keepquotes = !args->quoting;
		raw_csv->opts.doublequote = args->doublequote;
	} else {
		raw_csv->opts.quotechar = args->quotechar;
		raw_csv->opts.keepquotes = true;
		raw_csv->opts.doublequote = false;
	}
	raw_csv->opts.skipblanklines = args->skip_blank_lines;
	raw_csv->opts.comment = args->comment;
	raw_csv->opts.dayfirst = args->dayfirst;
	raw_csv->opts.decimal = args->decimal;
	raw_csv->opts.thousands = args->thousands;
	if (raw_csv->opts.decimal == raw_csv->opts.delimiter) {
		checkError(GDF_INVALID_API_CALL, "Decimal point cannot be the same as the delimiter");
	}
	if (raw_csv->opts.thousands == raw_csv->opts.delimiter) {
		checkError(GDF_INVALID_API_CALL, "Thousands separator cannot be the same as the delimiter");
	}

	string compression_type;
	error = inferCompressionType(args->compression, args->filepath_or_buffer, compression_type);
	checkError(error, "call to inferCompressionType");

	raw_csv->byte_range_offset = args->byte_range_offset;
	raw_csv->byte_range_size = args->byte_range_size;
	if (raw_csv->byte_range_offset > 0 || raw_csv->byte_range_size > 0) {
		if (raw_csv->nrows >= 0 || raw_csv->skiprows > 0 || raw_csv->skipfooter > 0) {
			checkError(GDF_INVALID_API_CALL, 
				"Cannot manually limit rows to be read when using the byte range parameter");
		}
		if (compression_type != "none") {
			checkError(GDF_INVALID_API_CALL, 
				"Cannot read compressed input when using the byte range parameter");
		}
	}

	// Handle user-defined booleans values, whereby field data is substituted
	// with true/false values; CUDF booleans are int types of 0 or 1
	// The true/false value strings are converted to integers which are used
	// by the data conversion kernel for comparison and value replacement
	if ((args->true_values != NULL) && (args->num_true_values > 0)) {
		thrust::host_vector<int32_t> h_values(args->num_true_values);
		for (int i = 0; i < args->num_true_values; ++i) {
			h_values[i] = convertStrToValue<int32_t>(args->true_values[i], 0, strlen(args->true_values[i]) - 1, raw_csv->opts);
		}
		raw_csv->d_trueValues = h_values;
	}
	if ((args->false_values != NULL) && (args->num_false_values > 0)) {
		thrust::host_vector<int32_t> h_values(args->num_false_values);
		for (int i = 0; i < args->num_false_values; ++i) {
			h_values[i] = convertStrToValue<int32_t>(args->false_values[i], 0, strlen(args->false_values[i]) - 1, raw_csv->opts);
		}
		raw_csv->d_falseValues = h_values;
	}

	if (args->na_filter && 
		(args->keep_default_na || (args->na_values != nullptr && args->num_na_values > 0))) {
		vector<string> na_values{
			"#N/A", "#N/A N/A", "#NA", "-1.#IND", 
			"-1.#QNAN", "-NaN", "-nan", "1.#IND", 
			"1.#QNAN", "N/A", "NA", "NULL", 
			"NaN", "n/a", "nan", "null"};
		if(!args->keep_default_na){
			na_values.clear();
		}

		if (args->na_values != nullptr && args->num_na_values > 0) {
			for (int i = 0; i < args->num_na_values; ++i) {
				na_values.emplace_back(args->na_values[i]);
			}
		}

		raw_csv->d_naTrie = createSerializedTrie(na_values);
	}

	raw_csv->opts.trueValues       = raw_csv->d_trueValues.data().get();
	raw_csv->opts.trueValuesCount  = raw_csv->d_trueValues.size();
	raw_csv->opts.falseValues      = raw_csv->d_falseValues.data().get();
	raw_csv->opts.falseValuesCount = raw_csv->d_falseValues.size();

	//-----------------------------------------------------------------------------
	// memory map in the data
	void * 	map_data = NULL;
	size_t	map_size = 0;
	size_t	map_offset = 0;
	int fd = 0;
	if (args->input_data_form == gdf_csv_input_form::FILE_PATH)
	{
		fd = open(args->filepath_or_buffer, O_RDONLY );
		if (fd < 0) 		{ close(fd); checkError(GDF_FILE_ERROR, "Error opening file"); }

		struct stat st{};
		if (fstat(fd, &st)) { close(fd); checkError(GDF_FILE_ERROR, "cannot stat file");   }
	
		const auto file_size = st.st_size;
		const auto page_size = sysconf(_SC_PAGESIZE);

		if (args->byte_range_offset >= (size_t)file_size) { 
			close(fd); 
			checkError(GDF_INVALID_API_CALL, "The byte_range offset is larger than the file size");
		}

		// Have to align map offset to page size
		map_offset = (args->byte_range_offset/page_size)*page_size;

		// Set to rest-of-the-file size, will reduce based on the byte range size
		raw_csv->num_bytes = map_size = file_size - map_offset;

		// Include the page padding in the mapped size
		const size_t page_padding = args->byte_range_offset - map_offset;
		const size_t padded_byte_range_size = raw_csv->byte_range_size + page_padding;

		if (raw_csv->byte_range_size != 0 && padded_byte_range_size < map_size) {
			// Need to make sure that w/ padding we don't overshoot the end of file
			map_size = min(padded_byte_range_size + calculateMaxRowSize(args->num_cols), map_size);
			// Ignore page padding for parsing purposes
			raw_csv->num_bytes = map_size - page_padding;
		}

		map_data = mmap(0, map_size, PROT_READ, MAP_PRIVATE, fd, map_offset);
	
		if (map_data == MAP_FAILED || map_size==0) { close(fd); checkError(GDF_C_ERROR, "Error mapping file"); }
	}
	else if (args->input_data_form == gdf_csv_input_form::HOST_BUFFER)
	{
		map_data = (void *)args->filepath_or_buffer;
		raw_csv->num_bytes = map_size = args->buffer_size;
	}
	else { checkError(GDF_C_ERROR, "invalid input type"); }

	const char* h_uncomp_data;
	size_t h_uncomp_size = 0;
	// Used when the input data is compressed, to ensure the allocated uncompressed data is freed
	vector<char> h_uncomp_data_owner;
	if (compression_type == "none") {
		// Do not use the owner vector here to avoid copying the whole file to the heap
		h_uncomp_data = (const char*)map_data + (args->byte_range_offset - map_offset);
		h_uncomp_size = raw_csv->num_bytes;
	}
	else {
		error = getUncompressedHostData( (const char *)map_data, map_size, compression_type, h_uncomp_data_owner);
		checkError(error, "call to getUncompressedHostData");
		h_uncomp_data = h_uncomp_data_owner.data();
		h_uncomp_size = h_uncomp_data_owner.size();
	}
	assert(h_uncomp_data != nullptr);
	assert(h_uncomp_size != 0);

	error = launch_countRecords(h_uncomp_data, h_uncomp_size, raw_csv, raw_csv->num_records);
	checkError(error, "call to record number of rows");

	//-----------------------------------------------------------------------------
	//-- Allocate space to hold the record starting point
	RMM_TRY( RMM_ALLOC(&raw_csv->recStart, sizeof(cu_recstart_t) * raw_csv->num_records, 0) ); 

	//-----------------------------------------------------------------------------
	//-- Scan data and set the starting positions
	error = launch_storeRecordStart(h_uncomp_data, h_uncomp_size, raw_csv);
	checkError(error, "call to record initial position store");

	// Previous kernel stores the record pinput_file.typeositions as encountered by all threads
	// Sort the record positions as subsequent processing may require filtering
	// certain rows or other processing on specific records
	thrust::sort(rmm::exec_policy()->on(0), raw_csv->recStart, raw_csv->recStart + raw_csv->num_records);

	// Currently, ignoring lineterminations within quotes is handled by recording
	// the records of both, and then filtering out the records that is a quotechar
	// or a linetermination within a quotechar pair. The future major refactoring
	// of csv_reader and its kernels will probably use a different tactic.
	if (raw_csv->opts.quotechar != '\0') {
		vector<cu_recstart_t> h_rec_starts(raw_csv->num_records);
		const size_t rec_start_size = sizeof(cu_recstart_t) * (h_rec_starts.size());
		CUDA_TRY( cudaMemcpy(h_rec_starts.data(), raw_csv->recStart, rec_start_size, cudaMemcpyDeviceToHost) );

		auto recCount = raw_csv->num_records;

		bool quotation = false;
		for (gdf_size_type i = 1; i < raw_csv->num_records; ++i) {
			if (h_uncomp_data[h_rec_starts[i] - 1] == raw_csv->opts.quotechar) {
				quotation = !quotation;
				h_rec_starts[i] = raw_csv->num_bytes;
				recCount--;
			}
			else if (quotation) {
				h_rec_starts[i] = raw_csv->num_bytes;
				recCount--;
			}
		}

		CUDA_TRY( cudaMemcpy(raw_csv->recStart, h_rec_starts.data(), rec_start_size, cudaMemcpyHostToDevice) );
		thrust::sort(rmm::exec_policy()->on(0), raw_csv->recStart, raw_csv->recStart + raw_csv->num_records);
		raw_csv->num_records = recCount;
	}

	error = uploadDataToDevice(h_uncomp_data, h_uncomp_size, raw_csv);
	if (error != GDF_SUCCESS) {
		return error;
	}

	//-----------------------------------------------------------------------------
	//-- Populate the header

	// Check if the user gave us a list of column names
	if(args->names == nullptr) {

		error = setColumnNamesFromCsv(raw_csv);
		if (error != GDF_SUCCESS) {
			return error;
		}
		const int h_num_cols = raw_csv->col_names.size();

		// Allocating a boolean array that will use to state if a column needs to read or filtered.
		raw_csv->h_parseCol = (bool*)malloc(sizeof(bool) * (h_num_cols));
		RMM_TRY( RMM_ALLOC((void**)&raw_csv->d_parseCol,(sizeof(bool) * (h_num_cols)),0 ) );
		for (int i = 0; i<h_num_cols; i++)
			raw_csv->h_parseCol[i]=true;
		
		// Rename empty column names to "Unnamed: col_index"
		for (size_t col_idx = 0; col_idx < raw_csv->col_names.size(); ++col_idx) {
			if (raw_csv->col_names[col_idx].empty()) {
				raw_csv->col_names[col_idx] = string("Unnamed: ") + std::to_string(col_idx);
			}
		}

		int h_dup_cols_removed = 0;
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

		raw_csv->num_actual_cols = h_num_cols;							// Actual number of columns in the CSV file
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


	//-----------------------------------------------------------------------------
	//---  done with host data
	if (args->input_data_form == gdf_csv_input_form::FILE_PATH)
	{
		close(fd);
		munmap(map_data, map_size);
	}


	//-----------------------------------------------------------------------------
	//--- Auto detect types of the vectors

	if(args->dtype==NULL){
		if (raw_csv->num_records == 0) {
			checkError(GDF_INVALID_API_CALL, "read_csv: no data available for data type inference");
		}

		column_data_t *d_ColumnData,*h_ColumnData;

		h_ColumnData = (column_data_t*)malloc(sizeof(column_data_t) * (raw_csv->num_active_cols));
		RMM_TRY( RMM_ALLOC((void**)&d_ColumnData,(sizeof(column_data_t) * (raw_csv->num_active_cols)),0 ) );

		CUDA_TRY( cudaMemset(d_ColumnData,	0, 	(sizeof(column_data_t) * (raw_csv->num_active_cols)) ) ) ;

		launch_dataTypeDetection(raw_csv, d_ColumnData);

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
    unsigned long long	*d_valid_count;
	gdf_dtype *d_dtypes,*h_dtypes;





	h_dtypes 		= (gdf_dtype*)malloc (	sizeof(gdf_dtype)* (raw_csv->num_active_cols));
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

	for (int acol = 0,col=-1; acol < raw_csv->num_actual_cols; acol++) {
		if(raw_csv->h_parseCol[acol]==false)
			continue;
		col++;

		gdf_column *gdf = (gdf_column *)malloc(sizeof(gdf_column) * 1);

		gdf->size		= raw_csv->num_records;
		gdf->dtype		= raw_csv->dtypes[col];
		gdf->null_count	= 0;						// will be filled in later

		//--- column name
		std::string str = raw_csv->col_names[acol];
		int len = str.length() + 1;
		gdf->col_name = (char *)malloc(sizeof(char) * len);
		memcpy(gdf->col_name, str.c_str(), len);
		gdf->col_name[len -1] = '\0';

		allocateGdfDataSpace(gdf);

		cols[col] 		= gdf;
		h_dtypes[col] 	= gdf->dtype;
		h_data[col] 	= gdf->data;
		h_valid[col] 	= gdf->valid;	
    }

	CUDA_TRY( cudaMemcpy(d_dtypes,h_dtypes, sizeof(gdf_dtype) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));
	CUDA_TRY( cudaMemcpy(d_data,h_data, sizeof(void*) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));
	CUDA_TRY( cudaMemcpy(d_valid,h_valid, sizeof(gdf_valid_type*) * (raw_csv->num_active_cols), cudaMemcpyHostToDevice));

	free(h_dtypes); 
	free(h_valid); 
	free(h_data); 

	if (raw_csv->num_records != 0) {
		error = launch_dataConvertColumns(raw_csv, d_data, d_valid, d_dtypes, d_str_cols, d_valid_count);
		if (error != GDF_SUCCESS) {
			return error;
		}
		// Sync with the default stream, just in case create_from_index() is asynchronous 
		cudaStreamSynchronize(0);

		stringColCount=0;
		for (int col = 0; col < raw_csv->num_active_cols; col++) {

			gdf_column *gdf = cols[col];

			if (gdf->dtype != gdf_dtype::GDF_STRING)
				continue;

			NVStrings* const stringCol = NVStrings::create_from_index(h_str_cols[stringColCount],size_t(raw_csv->num_records));
			if ((raw_csv->opts.quotechar != '\0') && (raw_csv->opts.doublequote==true)) {
				// In PANDAS, default of enabling doublequote for two consecutive
				// quotechar in quote fields results in reduction to single
				const string quotechar(1, raw_csv->opts.quotechar);
				const string doublequotechar(2, raw_csv->opts.quotechar);
				gdf->data = stringCol->replace(doublequotechar.c_str(), quotechar.c_str());
				NVStrings::destroy(stringCol);
			}
			else {
				gdf->data = stringCol;
			}

			RMM_TRY( RMM_FREE( h_str_cols [stringColCount], 0 ) );

			stringColCount++;
		}

		vector<unsigned long long>	h_valid_count(raw_csv->num_active_cols);
		CUDA_TRY( cudaMemcpy(h_valid_count.data(), d_valid_count, sizeof(unsigned long long) * h_valid_count.size(), cudaMemcpyDeviceToHost));

		//--- set the null count
		for (size_t col = 0; col < h_valid_count.size(); col++) {
			cols[col]->null_count = raw_csv->num_records - h_valid_count[col];
		}
	}

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
	RMM_TRY( RMM_FREE ( raw_csv->data, 0) );


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


/**---------------------------------------------------------------------------*
 * @brief Infer the compression type from the compression parameter and 
 * the input file name
 * 
 * Returns "none" if the input is not compressed.
 * 
 * @param[in] compression_arg Input string that is potentially describing 
 * the compression type. Can also be nullptr, "none", or "infer"
 * @param[in] filepath path + name of the input file
 * @param[out] compression_type String describing the inferred compression type
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error inferCompressionType(const char* compression_arg, const char* filepath, string& compression_type)
{
	if (compression_arg && 0 == strcasecmp(compression_arg, "none")) {
		compression_arg = nullptr;
	}
	if (compression_arg && 0 == strcasecmp(compression_arg, "infer"))
	{
		const char *file_ext = strrchr(filepath, '.');
		compression_arg = nullptr;
		if (file_ext)
		{
			if (!strcasecmp(file_ext, ".gz"))
				compression_arg = "gzip";
			else if (!strcasecmp(file_ext, ".zip"))
				compression_arg = "zip";
			else if (!strcasecmp(file_ext, ".bz2"))
				compression_arg = "bz2";
			else if (!strcasecmp(file_ext, ".xz"))
				compression_arg = "xz";
			else {
				// TODO: return error here
			}
		}
	}
	compression_type = compression_arg == nullptr? "none":string(compression_arg);
	
	return GDF_SUCCESS;
}


/**---------------------------------------------------------------------------*
 * @brief Uncompresses the input data and stores the allocated result into 
 * a vector.
 * 
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] num_bytes Size of the input data, in bytes
 * @param[in] compression String describing the compression type
 * @param[out] h_uncomp_data Vector containing the output uncompressed data
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error getUncompressedHostData(const char* h_data, size_t num_bytes, const string& compression, vector<char>& h_uncomp_data) 
{	
	int comp_type = IO_UNCOMP_STREAM_TYPE_INFER;
	if (compression == "gzip")
		comp_type = IO_UNCOMP_STREAM_TYPE_GZIP;
	else if (compression == "zip")
		comp_type = IO_UNCOMP_STREAM_TYPE_ZIP;
	else if (compression == "bz2")
		comp_type = IO_UNCOMP_STREAM_TYPE_BZIP2;
	else if (compression == "xz")
		comp_type = IO_UNCOMP_STREAM_TYPE_XZ;

	return io_uncompress_single_h2d(h_data, num_bytes, comp_type, h_uncomp_data);
}


/**---------------------------------------------------------------------------*
 * @brief Uploads the relevant segment of the input csv data onto the GPU.
 * 
 * Only rows that need to be read are copied to the GPU, based on parameters
 * like nrows, skipheader, skipfooter.
 * Also updates the array of record starts to match the device data offset.
 * 
 * @param[in] h_uncomp_data Pointer to the uncompressed csv data in host memory
 * @param[in] h_uncomp_size Size of the input data, in bytes
 * @param[in,out] raw_csv Structure containing the csv parsing parameters
 * and intermediate results
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error uploadDataToDevice(const char *h_uncomp_data, size_t h_uncomp_size,
                             raw_csv_t *raw_csv) {

  // Exclude the rows that are to be skipped from the start
  GDF_REQUIRE(raw_csv->num_records > raw_csv->skiprows, GDF_INVALID_API_CALL);
  const auto first_row = raw_csv->skiprows;
  raw_csv->num_records = raw_csv->num_records - first_row;

  std::vector<cu_recstart_t> h_rec_starts(raw_csv->num_records);
  CUDA_TRY(cudaMemcpy(h_rec_starts.data(), raw_csv->recStart + first_row,
                      sizeof(cu_recstart_t) * h_rec_starts.size(),
                      cudaMemcpyDefault));

  // Trim lines that are outside range, but keep one greater for the end offset
  if (raw_csv->byte_range_size != 0) {
    auto it = h_rec_starts.end() - 1;
    while (it >= h_rec_starts.begin() &&
           *it > cu_recstart_t(raw_csv->byte_range_size)) {
      --it;
    }
    if ((it + 2) < h_rec_starts.end()) {
      h_rec_starts.erase(it + 2, h_rec_starts.end());
    }
  }

  // Discard only blank lines, only fully comment lines, or both.
  // If only handling one of them, ensure it doesn't match against \0 as we do
  // not want certain scenarios to be filtered out (end-of-file)
  if (raw_csv->opts.skipblanklines || raw_csv->opts.comment != '\0') {
    const auto match1 = raw_csv->opts.skipblanklines ? raw_csv->opts.terminator
                                                     : raw_csv->opts.comment;
    const auto match2 = raw_csv->opts.comment != '\0' ? raw_csv->opts.comment
                                                      : match1;
    h_rec_starts.erase(
        std::remove_if(h_rec_starts.begin(), h_rec_starts.end(),
                       [&](cu_recstart_t i) {
                         return (h_uncomp_data[i] == match1 ||
                                 h_uncomp_data[i] == match2);
                       }),
        h_rec_starts.end());
  }

  raw_csv->num_records = h_rec_starts.size();

  // Exclude the rows before the header row (inclusive)
  // But copy the header data for parsing the column names later (if necessary)
  if (raw_csv->header_row >= 0) {
    raw_csv->header.assign(
        h_uncomp_data + h_rec_starts[raw_csv->header_row],
        h_uncomp_data + h_rec_starts[raw_csv->header_row + 1]);
    h_rec_starts.erase(h_rec_starts.begin(),
                       h_rec_starts.begin() + raw_csv->header_row + 1);
    raw_csv->num_records = h_rec_starts.size();
  }

  // Exclude the rows that exceed past the requested number
  if (raw_csv->nrows >= 0 && raw_csv->nrows < raw_csv->num_records) {
    h_rec_starts.resize(raw_csv->nrows + 1);    // include end offset
    raw_csv->num_records = h_rec_starts.size();
  }

  // Exclude the rows that are to be skipped from the end
  if (raw_csv->skipfooter > 0) {
    h_rec_starts.resize(h_rec_starts.size() - raw_csv->skipfooter);
    raw_csv->num_records = h_rec_starts.size();
  }

  // Check that there is actual data to parse
  GDF_REQUIRE(raw_csv->num_records > 0, GDF_INVALID_API_CALL);

  const auto start_offset = h_rec_starts.front();
  const auto end_offset = h_rec_starts.back();
  raw_csv->num_bytes = end_offset - start_offset;
  assert(raw_csv->num_bytes <= h_uncomp_size);
  raw_csv->num_bits = (raw_csv->num_bytes + 63) / 64;

  // Resize and upload the rows of interest
  RMM_TRY(RMM_REALLOC(&raw_csv->recStart,
                      sizeof(cu_recstart_t) * raw_csv->num_records, 0));
  CUDA_TRY(cudaMemcpy(raw_csv->recStart, h_rec_starts.data(),
                      sizeof(cu_recstart_t) * raw_csv->num_records,
                      cudaMemcpyDefault));

  // Upload the raw data that is within the rows of interest
  RMM_TRY(RMM_ALLOC(&raw_csv->data, raw_csv->num_bytes, 0));
  CUDA_TRY(cudaMemcpy(raw_csv->data, h_uncomp_data + start_offset,
                      raw_csv->num_bytes, cudaMemcpyHostToDevice));

  // Adjust row start positions to account for the data subcopy
  thrust::transform(rmm::exec_policy()->on(0), raw_csv->recStart,
                    raw_csv->recStart + raw_csv->num_records,
                    thrust::make_constant_iterator(start_offset),
                    raw_csv->recStart, thrust::minus<cu_recstart_t>());

  // The array of row offsets includes EOF
  // reduce the number of records by one to exclude it from the row count
  raw_csv->num_records--;

  return GDF_SUCCESS;
}


/**---------------------------------------------------------------------------*
 * @brief Allocates memory for a column's parsed output and its validity bitmap
 *
 * Memory for column data is simply based upon number of rows and the size of
 * the output data type, regardless of actual validity of the row element.
 *
 * @param[in,out] col The column whose memory will be allocated
 *
 * @return gdf_error GDF_SUCCESS upon completion
 *---------------------------------------------------------------------------**/
gdf_error allocateGdfDataSpace(gdf_column *col) {
  // TODO: We should not need to allocate space if there is nothing to parse
  // Need to debug/refactor the code to eliminate this requirement
  const auto num_rows = std::max(col->size, 1);
  const auto num_masks = gdf_get_num_chars_bitmask(num_rows);

  RMM_TRY(RMM_ALLOC(&col->valid, sizeof(gdf_valid_type) * num_masks, 0));
  CUDA_TRY(cudaMemset(col->valid, 0, sizeof(gdf_valid_type) * num_masks));

  if (col->dtype != gdf_dtype::GDF_STRING) {
    int column_byte_width = 0;
    checkError(get_column_byte_width(col, &column_byte_width),
               "Could not get column width using data type");
    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
  }

  return GDF_SUCCESS;
}

//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------


/**---------------------------------------------------------------------------*
 * @brief Counts the number of rows in the input csv file.
 * 
 * Does not load the entire file into the GPU memory at any time, so it can 
 * be used to parse large files.
 * Does not take quotes into consideration, so it will return extra rows
 * if the line terminating characters are present within quotes.
 * Because of this the result should be postprocessed to remove 
 * the fake line endings.
 * 
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in] terminator Line terminator character
 * @param[in] quote Quote character
 * @param[out] rec_cnt The resulting number of rows (records)
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error launch_countRecords(const char *h_data, size_t h_size,
                              raw_csv_t *raw_csv, gdf_size_type &rec_cnt)
{
	const size_t chunk_count = (h_size + max_chunk_bytes - 1) / max_chunk_bytes;
	rmm::device_vector<cu_reccnt_t> d_counts(chunk_count);

	char* d_chunk = nullptr;
	RMM_TRY(RMM_ALLOC (&d_chunk, max_chunk_bytes, 0)); 

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, countRecords));

	for (size_t ci = 0; ci < chunk_count; ++ci) {
		const auto h_chunk = h_data + ci * max_chunk_bytes;
		const auto chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);
		const auto chunk_bits = (chunk_bytes + 63) / 64;

		// Copy chunk to device
		CUDA_TRY(cudaMemcpy(d_chunk, h_chunk, chunk_bytes, cudaMemcpyDefault));

		const int gridSize = (chunk_bits + blockSize - 1) / blockSize;
		countRecords <<< gridSize, blockSize >>> (
			d_chunk, raw_csv->opts.terminator, raw_csv->opts.quotechar,
			chunk_bytes, chunk_bits, thrust::raw_pointer_cast(&d_counts[ci])
			);
	}

	RMM_TRY( RMM_FREE(d_chunk, 0) );

	CUDA_TRY(cudaGetLastError());

	// Row count is used to allocate/track row start positions
	// If not starting at an offset, add an extra row to account for offset=0
	rec_cnt = thrust::reduce(rmm::exec_policy()->on(0), d_counts.begin(), d_counts.end());
	if (raw_csv->byte_range_offset == 0) {
		rec_cnt++;
	}

	return GDF_SUCCESS;
}


/**---------------------------------------------------------------------------* 
 * @brief CUDA kernel that counts the number of rows in the given 
 * file segment, based on the location of line terminators. 
 * 
 * @param[in] data Device memory pointer to the csv data, 
 * potentially a chunk of the whole file
 * @param[in] terminator Line terminator character
 * @param[in] quotechar Quote character
 * @param[in] num_bytes Number of bytes in the input data
 * @param[in] num_bits Number of 'bits' in the input data. Each 'bit' is
 * processed by a separate CUDA thread
 * @param[in,out] num_records Device memory pointer to the number of found rows
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
__global__ void countRecords(char *data, const char terminator, const char quotechar, long num_bytes, long num_bits, 
	cu_reccnt_t* num_records) {

	// thread IDs range per block, so also need the block id
	const long tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if (tid >= num_bits)
		return;

	// data ID is a multiple of 64
	const long did = tid * 64L;

	const char *raw = (data + did);

	const long byteToProcess = ((did + 64L) < num_bytes) ? 64L : (num_bytes - did);

	// process the data
	cu_reccnt_t tokenCount = 0;
	for (long x = 0; x < byteToProcess; x++) {
		
		// Scan and log records. If quotations are enabled, then also log quotes
		// for a postprocess ignore, as the chunk here has limited visibility.
		if ((raw[x] == terminator) || (quotechar != '\0' && raw[x] == quotechar)) {
			tokenCount++;
		} else if (terminator == '\n' && (x + 1L) < byteToProcess && 
		           raw[x] == '\r' && raw[x + 1L] == '\n') {
			x++;
			tokenCount++;
		}

	}
	atomicAdd(num_records, tokenCount);
}


/**---------------------------------------------------------------------------*
 * @brief Finds the start of each row (record) in the given file, based on
 * the location of line terminators. The offset of each found row is stored 
 * in the recStart data member of the csvData parameter.
 * 
 * Does not load the entire file into the GPU memory at any time, so it can 
 * be used to parse large files.
 * Does not take quotes into consideration, so it will return extra rows
 * if the line terminating characters are present within quotes.
 * Because of this the result should be postprocessed to remove 
 * the fake line endings.
 * 
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in,out] csvData Structure containing the csv parsing parameters
 * and intermediate results
 * 
 * @return gdf_error with error code on failure, otherwise GDF_SUCCESS
 *---------------------------------------------------------------------------**/
gdf_error launch_storeRecordStart(const char *h_data, size_t h_size,
                                  raw_csv_t *csvData) {

	char* d_chunk = nullptr;
	// Allocate extra byte in case \r\n is at the chunk border
	RMM_TRY(RMM_ALLOC (&d_chunk, max_chunk_bytes + 1, 0)); 
	
	cu_reccnt_t*	d_num_records;
	RMM_TRY(RMM_ALLOC((void**)&d_num_records, sizeof(cu_reccnt_t), 0) );
	CUDA_TRY(cudaMemset(d_num_records, 0ull, sizeof(cu_reccnt_t)));

	int blockSize;		// suggested thread count to use
	int minGridSize;	// minimum block count required
	CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, storeRecordStart) );

	const size_t chunk_count = (h_size + max_chunk_bytes - 1) / max_chunk_bytes;
	for (size_t ci = 0; ci < chunk_count; ++ci) {	
		const auto chunk_offset = ci * max_chunk_bytes;	
		const auto h_chunk = h_data + chunk_offset;
		const auto chunk_bytes = std::min((size_t)(h_size - ci * max_chunk_bytes), max_chunk_bytes);
		const auto chunk_bits = (chunk_bytes + 63) / 64;
		// include_first_row should only apply to the first chunk
		const bool cu_include_first_row = (ci == 0) && (csvData->byte_range_offset == 0);
		
		// Copy chunk to device. Copy extra byte if not last chunk
		CUDA_TRY(cudaMemcpy(d_chunk, h_chunk, ci < (chunk_count - 1)?chunk_bytes:chunk_bytes + 1, cudaMemcpyDefault));

		const int gridSize = (chunk_bits + blockSize - 1) / blockSize;
		storeRecordStart <<< gridSize, blockSize >>> (
			d_chunk, chunk_offset, csvData->opts.terminator, csvData->opts.quotechar, cu_include_first_row,
			chunk_bytes, chunk_bits, d_num_records,
			csvData->recStart
		);
	}

	RMM_TRY( RMM_FREE( d_num_records, 0 ) ); 
	RMM_TRY( RMM_FREE( d_chunk, 0 ) );

	CUDA_TRY( cudaGetLastError() );

	return GDF_SUCCESS;
}


/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that finds the start of each row (record) in the given 
 * file segment, based on the location of line terminators. 
 * 
 * The offset of each found row is stored in a device memory array. 
 * The kernel operate on a segment (chunk) of the csv file.
 * 
 * @param[in] data Device memory pointer to the csv data, 
 * potentially a chunk of the whole file
 * @param[in] chunk_offset Offset of the data pointer from the start of the file
 * @param[in] terminator Line terminator character
 * @param[in] quotechar Quote character
 * @param[in] num_bytes Number of bytes in the input data
 * @param[in] num_bits Number of 'bits' in the input data. Each 'bit' is
 * processed by a separate CUDA thread
 * @param[in,out] num_records Device memory pointer to the number of found rows
 * @param[out] recStart device memory array containing the offset of each record
 * 
 * @return void
 *---------------------------------------------------------------------------**/
__global__ void storeRecordStart(char *data, size_t chunk_offset, 
	const char terminator, const char quotechar, bool include_first_row,
	long num_bytes, long num_bits, cu_reccnt_t* num_records,
	cu_recstart_t* recStart) {

	// thread IDs range per block, so also need the block id
	const long tid = threadIdx.x + (blockDim.x * blockIdx.x);

	if ( tid >= num_bits)
		return;

	// data ID - multiple of 64
	const long did = tid * 64L;

	if (did == 0 && include_first_row) {
		const auto pos = atomicAdd(num_records, 1ull);
		recStart[pos] = 0;
	}

	const char *raw = (data + did);

	const long byteToProcess = ((did + 64L) < num_bytes) ? 64L : (num_bytes - did);

	// process the data
	for (long x = 0; x < byteToProcess; x++) {

		// Scan and log records. If quotations are enabled, then also log quotes
		// for a postprocess ignore, as the chunk here has limited visibility.
		if ((raw[x] == terminator) || (quotechar != '\0' && raw[x] == quotechar)) {

			const auto pos = atomicAdd(num_records, 1ull);
			recStart[pos] = did + chunk_offset + x + 1;

		} else if (terminator == '\n' && (x + 1L) < byteToProcess && 
				   raw[x] == '\r' && raw[x + 1L] == '\n') {

			x++;
			const auto pos = atomicAdd(num_records, 1ull);
			recStart[pos] = did + chunk_offset + x + 1;
		}

	}
}


/**---------------------------------------------------------------------------*
 * @brief Helper function to setup and launch CSV parsing CUDA kernel.
 * 
 * @param[in,out] raw_csv The metadata for the CSV data
 * @param[out] gdf The output column data
 * @param[out] valid The bitmaps indicating whether column fields are valid
 * @param[out] str_cols The start/end offsets for string data types
 * @param[out] num_valid The numbers of valid fields in columns
 *
 * @return gdf_error GDF_SUCCESS upon completion
 *---------------------------------------------------------------------------**/
gdf_error launch_dataConvertColumns(raw_csv_t *raw_csv, void **gdf,
                                    gdf_valid_type **valid, gdf_dtype *d_dtypes,
                                    string_pair **str_cols,
                                    unsigned long long *num_valid) {
  int blockSize;    // suggested thread count to use
  int minGridSize;  // minimum block count required
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                              convertCsvToGdf));

  // Calculate actual block count to use based on records count
  int gridSize = (raw_csv->num_records + blockSize - 1) / blockSize;

  convertCsvToGdf <<< gridSize, blockSize >>> (
      raw_csv->data, raw_csv->opts, raw_csv->num_records,
      raw_csv->num_actual_cols, raw_csv->d_parseCol, raw_csv->recStart,
      d_dtypes,
      raw_csv->d_naTrie.empty() ? nullptr : raw_csv->d_naTrie.data().get(), gdf,
      valid, str_cols, num_valid);

  CUDA_TRY(cudaGetLastError());
  return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Functor for converting CSV data to cuDF data type value.
 *---------------------------------------------------------------------------**/
struct ConvertFunctor {
  /**---------------------------------------------------------------------------*
   * @brief Template specialization for operator() that handles integer types
   * that additionally checks whether the parsed data value should be overridden
   * with user-specified true/false matches.
   *
   * It is handled here rather than within convertStrToValue() as that function
   * is already used to construct the true/false match list from user-provided
   * strings at the start of parsing.
   *---------------------------------------------------------------------------**/
  template <typename T,
            typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ void operator()(
      const char *csvData, void *gdfColumnData, long rowIndex, long start,
      long end, const ParseOptions &opts) {
    T &value{static_cast<T *>(gdfColumnData)[rowIndex]};
    value = convertStrToValue<T>(csvData, start, end, opts);

    // Check for user-specified true/false values where the output is
    // replaced with 1/0 respectively
    if (isBooleanValue(value, opts.trueValues, opts.trueValuesCount)) {
      value = 1;
    } else if (isBooleanValue(value, opts.falseValues, opts.falseValuesCount)) {
      value = 0;
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Default template operator() dispatch specialization all data types
   * (including wrapper types) that is not covered by integral specialization.
   *---------------------------------------------------------------------------**/
  template <typename T,
            typename std::enable_if_t<!std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ void operator()(
      const char *csvData, void *gdfColumnData, long rowIndex, long start,
      long end, const ParseOptions &opts) {
    T &value{static_cast<T *>(gdfColumnData)[rowIndex]};
    value = convertStrToValue<T>(csvData, start, end, opts);
  }
};

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel iterates over the data until the end of the current field
 * 
 * Also iterates over (one or more) delimiter characters after the field.
 *
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] pos Offset to start the seeking from 
 * @param[in] stop Offset of the end of the row
 *
 * @return long position of the last character in the field, including the 
 *  delimiter(s) folloing the field data
 *---------------------------------------------------------------------------**/
__device__ 
long seekFieldEnd(const char *raw_csv, const ParseOptions opts, long pos, long stop) {
	bool quotation	= false;
	while(true){
		// Use simple logic to ignore control chars between any quote seq
		// Handles nominal cases including doublequotes within quotes, but
		// may not output exact failures as PANDAS for malformed fields
		if(raw_csv[pos] == opts.quotechar){
			quotation = !quotation;
		}
		else if(quotation==false){
			if(raw_csv[pos] == opts.delimiter){
				while (opts.multi_delimiter &&
					   pos < stop &&
					   raw_csv[pos + 1] == opts.delimiter) {
					++pos;
				}
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
	return pos;
}

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 * 
 * Data is processed one record at a time
 *
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] num_records The number of lines/rows of CSV data
 * @param[in] num_columns The number of columns of CSV data
 * @param[in] parseCol Whether to parse or skip a column
 * @param[in] recStart The start the CSV data of interest
 * @param[in] dtype The data type of the column
 * @param[out] gdf_data The output column data
 * @param[out] valid The bitmaps indicating whether column fields are valid
 * @param[out] str_cols The start/end offsets for string data types
 * @param[out] num_valid The numbers of valid fields in columns
 *
 * @return gdf_error GDF_SUCCESS upon completion
 *---------------------------------------------------------------------------**/
__global__
void convertCsvToGdf(char *raw_csv,
                     const ParseOptions opts,
                     gdf_size_type num_records,
                     int num_columns,
                     bool *parseCol,
                     cu_recstart_t *recStart,
                     gdf_dtype *dtype,
                     SerialTrieNode* na_trie,
                     void **gdf_data,
                     gdf_valid_type **valid,
                     string_pair **str_cols,
                     unsigned long long *num_valid)
{
	// thread IDs range per block, so also need the block id
	long	rec_id  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( rec_id >= num_records)
		return;

	long start 		= recStart[rec_id];
	long stop 		= recStart[rec_id + 1];

	long pos 		= start;
	int  col 		= 0;
	int  actual_col = 0;
	int  stringCol 	= 0;

	while(col<num_columns){

		if(start>stop)
			break;

		pos = seekFieldEnd(raw_csv, opts, pos, stop);

		if(parseCol[col]==true){

			// check if the entire field is a NaN string - consistent with pandas
			const bool is_na = (na_trie == nullptr) ? false : serializedTrieContains(na_trie, raw_csv + start, pos - start);

			// Modify start & end to ignore whitespace and quotechars
			long tempPos=pos-1;
			if(!is_na && dtype[actual_col] != gdf_dtype::GDF_CATEGORY && dtype[actual_col] != gdf_dtype::GDF_STRING){
				adjustForWhitespaceAndQuotes(raw_csv, &start, &tempPos, opts.quotechar);
			}

			if(!is_na && start<=(tempPos)) { // Empty fields are not legal values

				// Type dispatcher does not handle GDF_STRINGS
				if (dtype[actual_col] == gdf_dtype::GDF_STRING) {
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
				} else {
					cudf::type_dispatcher(
						dtype[actual_col], ConvertFunctor{}, raw_csv,
						gdf_data[actual_col], rec_id, start, tempPos, opts);
				}

				// set the valid bitmap - all bits were set to 0 to start
				int bitmapIdx 	= whichBitmap(rec_id);  	// which bitmap
				int bitIdx		= whichBit(rec_id);		// which bit - over an 8-bit index
				setBit(valid[actual_col]+bitmapIdx, bitIdx);		// This is done with atomics

				atomicAdd((unsigned long long int*)&num_valid[actual_col],(unsigned long long int)1);
			}
			else if(dtype[actual_col]==gdf_dtype::GDF_STRING){
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

/**---------------------------------------------------------------------------*
 * @brief Helper function to setup and launch CSV data type detect CUDA kernel.
 * 
 * @param[in] raw_csv The metadata for the CSV data
 * @param[out] d_columnData The count for each column data type
 *
 * @return gdf_error GDF_SUCCESS upon completion
 *---------------------------------------------------------------------------**/
gdf_error launch_dataTypeDetection(raw_csv_t *raw_csv,
                                   column_data_t *d_columnData) {
  int blockSize;    // suggested thread count to use
  int minGridSize;  // minimum block count required
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                              dataTypeDetection));

  // Calculate actual block count to use based on records count
  int gridSize = (raw_csv->num_records + blockSize - 1) / blockSize;

  dataTypeDetection <<< gridSize, blockSize >>> (
      raw_csv->data, raw_csv->opts, raw_csv->num_records,
      raw_csv->num_actual_cols, raw_csv->d_parseCol, raw_csv->recStart,
      d_columnData);

  CUDA_TRY(cudaGetLastError());
  return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @Param[in] raw_csv The entire CSV data to read
 * @Param[in] opts A set of parsing options
 * @Param[in] num_records The number of lines/rows of CSV data
 * @Param[in] num_columns The number of columns of CSV data
 * @Param[in] parseCol Whether to parse or skip a column
 * @Param[in] recStart The start the CSV data of interest
 * @Param[out] d_columnData The count for each column data type
 *
 * @Returns GDF_SUCCESS upon successful computation
 *---------------------------------------------------------------------------**/
__global__
void dataTypeDetection(char *raw_csv,
                       const ParseOptions opts,
                       gdf_size_type num_records,
                       int num_columns,
                       bool *parseCol,
                       cu_recstart_t *recStart,
                       column_data_t *d_columnData)
{
	// thread IDs range per block, so also need the block id
	long	rec_id  = threadIdx.x + (blockDim.x * blockIdx.x);		// this is entry into the field array - tid is an elements within the num_entries array

	// we can have more threads than data, make sure we are not past the end of the data
	if ( rec_id >= num_records)
		return;

	long start 		= recStart[rec_id];
	long stop 		= recStart[rec_id + 1];

	long pos 		= start;
	int  col 		= 0;
	int  actual_col = 0;

	// Going through all the columns of a given record
	while(col<num_columns){

		if(start>stop)
			break;

		pos = seekFieldEnd(raw_csv, opts, pos, stop);

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

			// Modify start & end to ignore whitespace and quotechars
			// This could possibly result in additional empty fields
			adjustForWhitespaceAndQuotes(raw_csv, &start, &tempPos);

			long strLen=tempPos-start+1;

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

			if(strLen==0){ // Removed spaces ' ' in the pre-processing and thus we can have an empty string.
				atomicAdd(& d_columnData[actual_col].countNULL, 1L);
			}
			// Integers have to have the length of the string or can be off by one if they start with a minus sign
			else if(countNumber==(strLen) || ( strLen>1 && countNumber==(strLen-1) && raw_csv[start]=='-') ){
				// Checking to see if we the integer value requires 8,16,32,64 bits.
				// This will allow us to allocate the exact amount of memory.
				const auto value = convertStrToValue<int64_t>(raw_csv, start, tempPos, opts);

				if (isBooleanValue<int32_t>(value, opts.trueValues, opts.trueValuesCount) ||
					isBooleanValue<int32_t>(value, opts.falseValues, opts.falseValuesCount)){
					atomicAdd(& d_columnData[actual_col].countInt8, 1L);
				}
				else if(value >= (1L<<31)){
					atomicAdd(& d_columnData[actual_col].countInt64, 1L);
				}
				else if(value >= (1L<<15)){
					atomicAdd(& d_columnData[actual_col].countInt32, 1L);
				}
				else if(value >= (1L<<7)){
					atomicAdd(& d_columnData[actual_col].countInt16, 1L);
				}
				else{
					atomicAdd(& d_columnData[actual_col].countInt8, 1L);
				}
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
