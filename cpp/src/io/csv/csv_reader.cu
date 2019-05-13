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
#include <memory>

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
#include "utilities/error_utils.hpp"
#include "utilities/trie.cuh"
#include "utilities/type_dispatcher.hpp"
#include "utilities/cudf_utils.h" 

#include <nvstrings/NVStrings.h>

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "io/comp/io_uncomp.h"

#include "io/cuio_common.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/wrapper_utils.hpp"

using std::vector;
using std::string;

/**---------------------------------------------------------------------------*
 * @brief Struct used for internal parsing state
 *---------------------------------------------------------------------------**/
typedef struct raw_csv_ {
    device_buffer<char> 	data;		// on-device: the raw unprocessed CSV data - loaded as a large char * array
    device_buffer<uint64_t> recStart;	// on-device: Starting position of the records.

    ParseOptions			opts;			// options to control parsing behavior

    long				num_bytes;		// host: the number of bytes in the data
    long				num_bits;		// host: the number of 64-bit bitmaps (different than valid)
	gdf_size_type 		num_records;	// host: number of records loaded into device memory, and then number of records to read
	// int				num_cols;		// host: number of columns
	int					num_active_cols;// host: number of columns that will be return to user.
	int					num_actual_cols;// host: number of columns in the file --- based on the number of columns in header
    vector<gdf_dtype>	dtypes;			// host: array of dtypes (since gdf_columns are not created until end)
    vector<string>		col_names;		// host: array of column names
	
	thrust::host_vector<bool>	h_parseCol;	// host   : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.
    rmm::device_vector<bool>	d_parseCol;	// device : array of booleans stating if column should be parsed in reading process: parseCol[x]=false means that the column x needs to be filtered out.

    long        byte_range_offset;  // offset into the data to start parsing
    long        byte_range_size;    // length of the data of interest to parse

    gdf_size_type header_row;       ///< host: Row index of the header
    gdf_size_type nrows;            ///< host: Number of rows to read. -1 for all rows
    gdf_size_type skiprows;         ///< host: Number of rows to skip from the start
    gdf_size_type skipfooter;       ///< host: Number of rows to skip from the end
    std::vector<char> header;       ///< host: Header row data, for parsing column names
    string prefix;                  ///< host: Prepended to column ID if there is no header or input column names

    rmm::device_vector<SerialTrieNode>	d_trueTrie;	// device: serialized trie of values to recognize as true
    rmm::device_vector<SerialTrieNode>	d_falseTrie;// device: serialized trie of values to recognize as false
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

#define checkError(error, txt)  if ( error != GDF_SUCCESS) { std::cerr << "ERROR:  " << error <<  "  in "  << txt << std::endl;  return error; }

//
//---------------CUDA Kernel ---------------------------------------------
//

gdf_error launch_dataConvertColumns(raw_csv_t *raw_csv, void **d_gdf,
                                    gdf_valid_type **valid, gdf_dtype *d_dtypes,
                                    gdf_size_type *num_valid);
gdf_error launch_dataTypeDetection(raw_csv_t *raw_csv,
                                   column_data_t *d_columnData);

__global__ void convertCsvToGdf(char *csv, const ParseOptions opts,
                                gdf_size_type num_records, int num_columns,
                                bool *parseCol, uint64_t *recStart,
                                gdf_dtype *dtype, void **gdf_data,
                                gdf_valid_type **valid,
                                gdf_size_type *num_valid);
__global__ void dataTypeDetection(char *raw_csv, const ParseOptions opts,
                                  gdf_size_type num_records, int num_columns,
                                  bool *parseCol, uint64_t *recStart,
                                  column_data_t *d_columnData);

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
		uint64_t first_row_len{};
		// If file only contains one row, raw_csv->recStart[1] is not valid
		if (raw_csv->num_records > 1) {
			CUDA_TRY(cudaMemcpy(&first_row_len, raw_csv->recStart.data() + 1, sizeof(uint64_t), cudaMemcpyDefault));
		}
		else {
			// File has one row - use the file size for the row size
			first_row_len = raw_csv->num_bytes / sizeof(char);
		}
		first_row.resize(first_row_len);
		CUDA_TRY(cudaMemcpy(first_row.data(), raw_csv->data.data(), first_row_len * sizeof(char), cudaMemcpyDefault));
	}

	int num_cols = 0;

	bool quotation	= false;
	for (size_t pos = 0, prev = 0; pos < first_row.size(); ++pos) {
		// Flip the quotation flag if current character is a quotechar
		if(first_row[pos] == raw_csv->opts.quotechar) {
			quotation = !quotation;
		}
		// Check if end of a column/row
		else if (pos == first_row.size() - 1 ||
				 (!quotation && first_row[pos] == raw_csv->opts.terminator) ||
				 (!quotation && first_row[pos] == raw_csv->opts.delimiter)) {
			// This is the header, add the column name
			if (raw_csv->header_row >= 0) {
				// Include the current character, in case the line is not terminated
				int col_name_len = pos - prev + 1;
				// Exclude the delimiter/terminator is present
				if (first_row[pos] == raw_csv->opts.delimiter || first_row[pos] == raw_csv->opts.terminator) {
					--col_name_len;
				}
				// Also exclude '\r' character at the end of the column name if it's part of the terminator
				if (col_name_len > 0 &&
					raw_csv->opts.terminator == '\n' &&
					first_row[pos] == '\n' &&
					first_row[pos - 1] == '\r') {
					--col_name_len;
				}

				const string new_col_name(first_row.data() + prev, col_name_len);
				raw_csv->col_names.push_back(removeQuotes(new_col_name, raw_csv->opts.quotechar));

				// Stop parsing when we hit the line terminator; relevant when there is a blank line following the header.
				// In this case, first_row includes multiple line terminators at the end, as the new recStart belongs
				// to a line that comes after the blank line(s)
				if (!quotation && first_row[pos] == raw_csv->opts.terminator){
					break;
				}
			}
			else {
				// This is the first data row, add the automatically generated name
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
 * @brief Updates the raw_csv_t object with the total number of rows and
 * quotation characters in the file
 *
 * Does not count the quotations if quotechar is set to '/0'.
 *
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in,out] raw_csv Structure containing the csv parsing parameters
 * and intermediate results
 *
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error countRecordsAndQuotes(const char *h_data, size_t h_size, raw_csv_t *raw_csv) {
	vector<char> chars_to_count{raw_csv->opts.terminator};
	if (raw_csv->opts.quotechar != '\0') {
		chars_to_count.push_back(raw_csv->opts.quotechar);
	}

	raw_csv->num_records = countAllFromSet(h_data, h_size, chars_to_count);

	// If not starting at an offset, add an extra row to account for the first row in the file
	if (raw_csv->byte_range_offset == 0) {
		++raw_csv->num_records;
	}

	return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Updates the raw_csv_t object with the offset of each row in the file
 * Also add positions of each quotation character in the file.
 *
 * Does not process the quotations if quotechar is set to '/0'.
 *
 * @param[in] h_data Pointer to the csv data in host memory
 * @param[in] h_size Size of the input data, in bytes
 * @param[in,out] raw_csv Structure containing the csv parsing parameters
 * and intermediate results
 *
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error setRecordStarts(const char *h_data, size_t h_size, raw_csv_t *raw_csv) {
	// Allocate space to hold the record starting points
	const bool last_line_terminated = (h_data[h_size - 1] == raw_csv->opts.terminator);
	// If the last line is not terminated, allocate space for the EOF entry (added later)
	const gdf_size_type record_start_count = raw_csv->num_records + (last_line_terminated ? 0 : 1);
	raw_csv->recStart = device_buffer<uint64_t>(record_start_count); 

	auto* find_result_ptr = raw_csv->recStart.data();
	if (raw_csv->byte_range_offset == 0) {
		find_result_ptr++;
		CUDA_TRY(cudaMemsetAsync(raw_csv->recStart.data(), 0ull, sizeof(uint64_t)));
	}
	vector<char> chars_to_find{raw_csv->opts.terminator};
	if (raw_csv->opts.quotechar != '\0') {
		chars_to_find.push_back(raw_csv->opts.quotechar);
	}
	// Passing offset = 1 to return positions AFTER the found character
	findAllFromSet(h_data, h_size, chars_to_find, 1, find_result_ptr);

	// Previous call stores the record pinput_file.typeositions as encountered by all threads
	// Sort the record positions as subsequent processing may require filtering
	// certain rows or other processing on specific records
	thrust::sort(rmm::exec_policy()->on(0), raw_csv->recStart.data(), raw_csv->recStart.data() + raw_csv->num_records);

	// Currently, ignoring lineterminations within quotes is handled by recording
	// the records of both, and then filtering out the records that is a quotechar
	// or a linetermination within a quotechar pair. The future major refactoring
	// of csv_reader and its kernels will probably use a different tactic.
	if (raw_csv->opts.quotechar != '\0') {
		vector<uint64_t> h_rec_starts(raw_csv->num_records);
		const size_t rec_start_size = sizeof(uint64_t) * (h_rec_starts.size());
		CUDA_TRY( cudaMemcpy(h_rec_starts.data(), raw_csv->recStart.data(), rec_start_size, cudaMemcpyDeviceToHost) );

		auto recCount = raw_csv->num_records;

		bool quotation = false;
		for (gdf_size_type i = 1; i < raw_csv->num_records; ++i) {
			if (h_data[h_rec_starts[i] - 1] == raw_csv->opts.quotechar) {
				quotation = !quotation;
				h_rec_starts[i] = raw_csv->num_bytes;
				recCount--;
			}
			else if (quotation) {
				h_rec_starts[i] = raw_csv->num_bytes;
				recCount--;
			}
		}

		CUDA_TRY( cudaMemcpy(raw_csv->recStart.data(), h_rec_starts.data(), rec_start_size, cudaMemcpyHostToDevice) );
		thrust::sort(rmm::exec_policy()->on(0), raw_csv->recStart.data(), raw_csv->recStart.data() + raw_csv->num_records);
		raw_csv->num_records = recCount;
	}

	if (!last_line_terminated){
		// Add the EOF as the last record when the terminator is missing in the last line
		const uint64_t eof_offset = h_size;
		CUDA_TRY(cudaMemcpy(raw_csv->recStart.data() + raw_csv->num_records, &eof_offset, sizeof(uint64_t), cudaMemcpyDefault));
		// Update the record count
		++raw_csv->num_records;
	}

	return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Reads CSV-structured data and returns an array of gdf_columns.
 *
 * @param[in,out] args Structure containing input and output args
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 *---------------------------------------------------------------------------**/
gdf_error read_csv(csv_read_arg *args)
{
  gdf_error error = gdf_error::GDF_SUCCESS;

	//-----------------------------------------------------------------------------
	// create the CSV data structure - this will be filled in as the CSV data is processed.
	// Done first to validate data types
	raw_csv_t raw_csv{};
	// error = parseArguments(args, raw_csv);
	raw_csv.num_actual_cols	= args->num_cols;
	raw_csv.num_active_cols	= args->num_cols;
	raw_csv.num_records		= 0;

	raw_csv.header_row = args->header;
	raw_csv.skiprows = args->skiprows;
	raw_csv.skipfooter = args->skipfooter;
	raw_csv.nrows = args->nrows;
	raw_csv.prefix = args->prefix == nullptr ? "" : string(args->prefix);

	if (args->delim_whitespace) {
		raw_csv.opts.delimiter = ' ';
		raw_csv.opts.multi_delimiter = true;
	} else {
		raw_csv.opts.delimiter = args->delimiter;
		raw_csv.opts.multi_delimiter = false;
	}
	if (args->windowslinetermination) {
		raw_csv.opts.terminator = '\n';
	} else {
		raw_csv.opts.terminator = args->lineterminator;
	}
	if (args->quotechar != '\0' && args->quoting != QUOTE_NONE) {
		raw_csv.opts.quotechar = args->quotechar;
		raw_csv.opts.keepquotes = false;
		raw_csv.opts.doublequote = args->doublequote;
	} else {
		raw_csv.opts.quotechar = '\0';
		raw_csv.opts.keepquotes = true;
		raw_csv.opts.doublequote = false;
	}
	raw_csv.opts.skipblanklines = args->skip_blank_lines;
	raw_csv.opts.comment = args->comment;
	raw_csv.opts.dayfirst = args->dayfirst;
	raw_csv.opts.decimal = args->decimal;
	raw_csv.opts.thousands = args->thousands;
	if (raw_csv.opts.decimal == raw_csv.opts.delimiter) {
		checkError(GDF_INVALID_API_CALL, "Decimal point cannot be the same as the delimiter");
	}
	if (raw_csv.opts.thousands == raw_csv.opts.delimiter) {
		checkError(GDF_INVALID_API_CALL, "Thousands separator cannot be the same as the delimiter");
	}

	string compression_type;
	error = inferCompressionType(args->compression, args->filepath_or_buffer, compression_type);
	checkError(error, "call to inferCompressionType");

	raw_csv.byte_range_offset = args->byte_range_offset;
	raw_csv.byte_range_size = args->byte_range_size;
	if (raw_csv.byte_range_offset > 0 || raw_csv.byte_range_size > 0) {
		if (raw_csv.nrows >= 0 || raw_csv.skiprows > 0 || raw_csv.skipfooter > 0) {
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
	vector<string> true_values{"True", "TRUE"};
	if (args->true_values != nullptr && args->num_true_values > 0) {
		for (int i = 0; i < args->num_true_values; ++i) {
			true_values.emplace_back(args->true_values[i]);
		}
	}
	raw_csv.d_trueTrie = createSerializedTrie(true_values);
	raw_csv.opts.trueValuesTrie = raw_csv.d_trueTrie.data().get();

	vector<string> false_values{"False", "FALSE"};
	if (args->false_values != nullptr && args->num_false_values > 0) {
		for (int i = 0; i < args->num_false_values; ++i) {
			false_values.emplace_back(args->false_values[i]);
		}
	}
	raw_csv.d_falseTrie = createSerializedTrie(false_values);
	raw_csv.opts.falseValuesTrie = raw_csv.d_falseTrie.data().get();

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

		raw_csv.d_naTrie = createSerializedTrie(na_values);
		raw_csv.opts.naValuesTrie = raw_csv.d_naTrie.data().get();
	}
	args->data = nullptr;

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
		raw_csv.num_bytes = map_size = file_size - map_offset;

		// Include the page padding in the mapped size
		const size_t page_padding = args->byte_range_offset - map_offset;
		const size_t padded_byte_range_size = raw_csv.byte_range_size + page_padding;

		if (raw_csv.byte_range_size != 0 && padded_byte_range_size < map_size) {
			// Need to make sure that w/ padding we don't overshoot the end of file
			map_size = min(padded_byte_range_size + calculateMaxRowSize(args->num_cols), map_size);

		}

		// Ignore page padding for parsing purposes
		raw_csv.num_bytes = map_size - page_padding;

		map_data = mmap(0, map_size, PROT_READ, MAP_PRIVATE, fd, map_offset);
	
		if (map_data == MAP_FAILED || map_size==0) { close(fd); checkError(GDF_C_ERROR, "Error mapping file"); }
	}
	else if (args->input_data_form == gdf_csv_input_form::HOST_BUFFER)
	{
		map_data = (void *)args->filepath_or_buffer;
		raw_csv.num_bytes = map_size = args->buffer_size;
	}
	else { checkError(GDF_C_ERROR, "invalid input type"); }

	const char* h_uncomp_data;
	size_t h_uncomp_size = 0;
	// Used when the input data is compressed, to ensure the allocated uncompressed data is freed
	vector<char> h_uncomp_data_owner;
	if (compression_type == "none") {
		// Do not use the owner vector here to avoid copying the whole file to the heap
		h_uncomp_data = (const char*)map_data + (args->byte_range_offset - map_offset);
		h_uncomp_size = raw_csv.num_bytes;
	}
	else {
		error = getUncompressedHostData( (const char *)map_data, map_size, compression_type, h_uncomp_data_owner);
		checkError(error, "call to getUncompressedHostData");
		h_uncomp_data = h_uncomp_data_owner.data();
		h_uncomp_size = h_uncomp_data_owner.size();
	}
	assert(h_uncomp_data != nullptr);
	assert(h_uncomp_size != 0);

	error = countRecordsAndQuotes(h_uncomp_data, h_uncomp_size, &raw_csv);
	checkError(error, "call to count the number of rows");

	error = setRecordStarts(h_uncomp_data, h_uncomp_size, &raw_csv);
	checkError(error, "call to store the row offsets");

	error = uploadDataToDevice(h_uncomp_data, h_uncomp_size, &raw_csv);
	checkError(error, "call to upload the CSV data to the device");

	//-----------------------------------------------------------------------------
	//---  done with host data
	if (args->input_data_form == gdf_csv_input_form::FILE_PATH)
	{
		close(fd);
		munmap(map_data, map_size);
	}

	//-----------------------------------------------------------------------------
	//-- Populate the header

	// Check if the user gave us a list of column names
	if(args->names == nullptr) {

		error = setColumnNamesFromCsv(&raw_csv);
		if (error != GDF_SUCCESS) {
			return error;
		}
		const int h_num_cols = raw_csv.col_names.size();

		// Initialize a boolean array that states if a column needs to read or filtered.
		raw_csv.h_parseCol = thrust::host_vector<bool>(h_num_cols, true);
		
		// Rename empty column names to "Unnamed: col_index"
		for (size_t col_idx = 0; col_idx < raw_csv.col_names.size(); ++col_idx) {
			if (raw_csv.col_names[col_idx].empty()) {
				raw_csv.col_names[col_idx] = string("Unnamed: ") + std::to_string(col_idx);
			}
		}

		int h_dup_cols_removed = 0;
		// Looking for duplicates
		for (auto it = raw_csv.col_names.begin(); it != raw_csv.col_names.end(); it++){
			bool found_dupe = false;
			for (auto it2 = (it+1); it2 != raw_csv.col_names.end(); it2++){
				if (*it==*it2){
					found_dupe=true;
					break;
				}
			}
			if(found_dupe){
				int count=1;
				for (auto it2 = (it+1); it2 != raw_csv.col_names.end(); it2++){
					if (*it==*it2){
						if(args->mangle_dupe_cols){
							// Replace all the duplicates of column X with X.1,X.2,... First appearance stays as X.
							std::string newColName  = *it2;
							newColName += "." + std::to_string(count); 
							count++;
							*it2 = newColName;							
						} else{
							// All duplicate fields will be ignored.
							int pos=std::distance(raw_csv.col_names.begin(), it2);
							raw_csv.h_parseCol[pos]=false;
							h_dup_cols_removed++;
						}
					}
				}
			}
		}

		raw_csv.num_actual_cols = h_num_cols;							// Actual number of columns in the CSV file
		raw_csv.num_active_cols = h_num_cols-h_dup_cols_removed;		// Number of fields that need to be processed based on duplicatation fields

	}
	else {
		raw_csv.h_parseCol = thrust::host_vector<bool>(args->num_cols, true);

		for (int i = 0; i<raw_csv.num_actual_cols; i++){
			std::string col_name 	= args->names[i];
			raw_csv.col_names.push_back(col_name);
		}
	}

	// User can give
	if (args->use_cols_int!=NULL || args->use_cols_char!=NULL){
		if(args->use_cols_int!=NULL){
			for (int i = 0; i<raw_csv.num_actual_cols; i++)
				raw_csv.h_parseCol[i]=false;
			for(int i=0; i < args->use_cols_int_len; i++){
				int pos = args->use_cols_int[i];
				raw_csv.h_parseCol[pos]=true;
			}
			raw_csv.num_active_cols = args->use_cols_int_len;
		}else{
			for (int i = 0; i<raw_csv.num_actual_cols; i++)
				raw_csv.h_parseCol[i]=false;
			int countFound=0;
			for(int i=0; i < args->use_cols_char_len; i++){
				std::string colName(args->use_cols_char[i]);
				for (auto it = raw_csv.col_names.begin(); it != raw_csv.col_names.end(); it++){
					if(colName==*it){
						countFound++;
						int pos=std::distance(raw_csv.col_names.begin(), it);
						raw_csv.h_parseCol[pos]=true;
						break;
					}
				}
			}
			raw_csv.num_active_cols = countFound;
		}
	}
	raw_csv.d_parseCol = raw_csv.h_parseCol;

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
		if (raw_csv.num_records == 0) {
			checkError(GDF_INVALID_API_CALL, "read_csv: no data available for data type inference");
		}

		vector<column_data_t> h_ColumnData(raw_csv.num_active_cols);
		device_buffer<column_data_t> d_ColumnData(raw_csv.num_active_cols);
		CUDA_TRY( cudaMemset(d_ColumnData.data(),	0, 	(sizeof(column_data_t) * (raw_csv.num_active_cols)) ) ) ;

		launch_dataTypeDetection(&raw_csv, d_ColumnData.data());
		CUDA_TRY( cudaMemcpy(h_ColumnData.data(), d_ColumnData.data(), sizeof(column_data_t) * (raw_csv.num_active_cols), cudaMemcpyDeviceToHost));

		// host: array of dtypes (since gdf_columns are not created until end)
		vector<gdf_dtype>	d_detectedTypes;

		raw_csv.dtypes.clear();

		for(int col = 0; col < raw_csv.num_active_cols; col++){
			unsigned long long countInt = h_ColumnData[col].countInt8+h_ColumnData[col].countInt16+
										  h_ColumnData[col].countInt32+h_ColumnData[col].countInt64;

			if (h_ColumnData[col].countNULL == raw_csv.num_records){
				d_detectedTypes.push_back(GDF_INT8); // Entire column is NULL. Allocating the smallest amount of memory
			} else if(h_ColumnData[col].countString>0L){
				d_detectedTypes.push_back(GDF_STRING); // For auto-detection, we are currently not supporting strings.
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
		raw_csv.dtypes=d_detectedTypes;
	}
	else{
		for ( int x = 0; x < raw_csv.num_actual_cols; x++) {

			std::string temp_type 	= args->dtype[x];
                        gdf_dtype col_dtype = GDF_invalid;
			if(temp_type.find(':') != std::string::npos){
				for (auto it = raw_csv.col_names.begin(); it != raw_csv.col_names.end(); it++){
				std::size_t idx = temp_type.find(':');
				if(temp_type.substr( 0, idx) == *it){
					std::string temp_dtype = temp_type.substr( idx +1);
					col_dtype	= convertStringToDtype(temp_dtype);
					break;
					}
				}
			}
			else{
				col_dtype	= convertStringToDtype( temp_type );
			}

			if (col_dtype == GDF_invalid)
				return GDF_UNSUPPORTED_DTYPE;

			raw_csv.dtypes.push_back(col_dtype);
		}
	}

  // Alloc output; columns' data memory is still expected for empty dataframe
  std::vector<gdf_column_wrapper> columns;
  for (int col = 0, active_col = 0; col < raw_csv.num_actual_cols; ++col) {
    if (raw_csv.h_parseCol[col]) {
      // When dtypes are inferred, it contains only active column values
      auto dtype = raw_csv.dtypes[args->dtype == nullptr ? active_col : col];

      columns.emplace_back(raw_csv.num_records, dtype,
                           gdf_dtype_extra_info{TIME_UNIT_NONE},
                           raw_csv.col_names[col]);
      CUDF_EXPECTS(columns.back().allocate() == GDF_SUCCESS, "Cannot allocate columns");
      active_col++;
    }
  }

  // Convert CSV input to cuDF output
  if (raw_csv.num_records != 0) {
    thrust::host_vector<gdf_dtype> h_dtypes(raw_csv.num_active_cols);
    thrust::host_vector<void*> h_data(raw_csv.num_active_cols);
    thrust::host_vector<gdf_valid_type*> h_valid(raw_csv.num_active_cols);

    for (int i = 0; i < raw_csv.num_active_cols; ++i) {
      h_dtypes[i] = columns[i]->dtype;
      h_data[i] = columns[i]->data;
      h_valid[i] = columns[i]->valid;
    }

    rmm::device_vector<gdf_dtype> d_dtypes = h_dtypes;
    rmm::device_vector<void*> d_data = h_data;
    rmm::device_vector<gdf_valid_type*> d_valid = h_valid;
    rmm::device_vector<gdf_size_type> d_valid_counts(raw_csv.num_active_cols, 0);

    CUDF_EXPECTS(
        launch_dataConvertColumns(&raw_csv, d_data.data().get(),
                                  d_valid.data().get(), d_dtypes.data().get(),
                                  d_valid_counts.data().get()) == GDF_SUCCESS,
        "Cannot convert CSV data to cuDF columns");
    CUDA_TRY(cudaStreamSynchronize(0));

    thrust::host_vector<gdf_size_type> h_valid_counts = d_valid_counts;
    for (int i = 0; i < raw_csv.num_active_cols; ++i) {
      columns[i]->null_count = columns[i]->size - h_valid_counts[i];
    }
  }

  for (int i = 0; i < raw_csv.num_active_cols; ++i) {
    if (columns[i]->dtype == GDF_STRING) {
      std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)> str_data(
        NVStrings::create_from_index(static_cast<string_pair *>(columns[i]->data), columns[i]->size), 
        &NVStrings::destroy);
      RMM_TRY(RMM_FREE(columns[i]->data, 0));

      // PANDAS' default behavior of enabling doublequote for two consecutive
      // quotechars in quoted fields results in reduction to a single quotechar
      if ((raw_csv.opts.quotechar != '\0') &&
          (raw_csv.opts.doublequote == true)) {
        const std::string quotechar(1, raw_csv.opts.quotechar);
        const std::string doublequotechar(2, raw_csv.opts.quotechar);
        columns[i]->data = str_data->replace(doublequotechar.c_str(), quotechar.c_str());
      }
      else {
        columns[i]->data = str_data.release();
      }
    }
  }

  // Transfer ownership to raw pointer output arguments
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * raw_csv.num_active_cols);
  for (int i = 0; i < raw_csv.num_active_cols; ++i) {
    args->data[i] = columns[i].release();
  }
  args->num_cols_out = raw_csv.num_active_cols;
  args->num_rows_out = raw_csv.num_records;

  return error;
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

  std::vector<uint64_t> h_rec_starts(raw_csv->num_records);
  CUDA_TRY(cudaMemcpy(h_rec_starts.data(), raw_csv->recStart.data() + first_row,
                      sizeof(uint64_t) * h_rec_starts.size(),
                      cudaMemcpyDefault));

  // Trim lines that are outside range, but keep one greater for the end offset
  if (raw_csv->byte_range_size != 0) {
    auto it = h_rec_starts.end() - 1;
    while (it >= h_rec_starts.begin() &&
           *it > uint64_t(raw_csv->byte_range_size)) {
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
    const auto match_newline = raw_csv->opts.skipblanklines ? raw_csv->opts.terminator
                                                            : raw_csv->opts.comment;
    const auto match_comment = raw_csv->opts.comment != '\0' ? raw_csv->opts.comment
                                                             : match_newline;
    const auto match_return = (raw_csv->opts.skipblanklines &&
                              raw_csv->opts.terminator == '\n') ? '\r'
                                                                : match_comment;
    h_rec_starts.erase(
        std::remove_if(h_rec_starts.begin(), h_rec_starts.end(),
                       [&](uint64_t i) {
                         return (h_uncomp_data[i] == match_newline ||
                                 h_uncomp_data[i] == match_return ||
                                 h_uncomp_data[i] == match_comment);
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
  raw_csv->recStart.resize(raw_csv->num_records);
  CUDA_TRY(cudaMemcpy(raw_csv->recStart.data(), h_rec_starts.data(),
                      sizeof(uint64_t) * raw_csv->num_records,
                      cudaMemcpyDefault));

  // Upload the raw data that is within the rows of interest
  raw_csv->data = device_buffer<char>(raw_csv->num_bytes);
  CUDA_TRY(cudaMemcpy(raw_csv->data.data(), h_uncomp_data + start_offset,
                      raw_csv->num_bytes, cudaMemcpyHostToDevice));

  // Adjust row start positions to account for the data subcopy
  thrust::transform(rmm::exec_policy()->on(0), raw_csv->recStart.data(),
                    raw_csv->recStart.data() + raw_csv->num_records,
                    thrust::make_constant_iterator(start_offset),
                    raw_csv->recStart.data(), thrust::minus<uint64_t>());

  // The array of row offsets includes EOF
  // reduce the number of records by one to exclude it from the row count
  raw_csv->num_records--;

  return GDF_SUCCESS;
}

//----------------------------------------------------------------------------------------------------------------
//				CUDA Kernels
//----------------------------------------------------------------------------------------------------------------

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
                                    gdf_size_type *num_valid) {
  int blockSize;    // suggested thread count to use
  int minGridSize;  // minimum block count required
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                              convertCsvToGdf));

  // Calculate actual block count to use based on records count
  int gridSize = (raw_csv->num_records + blockSize - 1) / blockSize;

  convertCsvToGdf <<< gridSize, blockSize >>> (
      raw_csv->data.data(), raw_csv->opts, raw_csv->num_records,
      raw_csv->num_actual_cols, raw_csv->d_parseCol.data().get(), raw_csv->recStart.data(),
      d_dtypes, gdf, valid, num_valid);

  CUDA_TRY(cudaGetLastError());
  return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Functor for converting CSV data to cuDF data type value.
 *---------------------------------------------------------------------------**/
struct ConvertFunctor {
  /**---------------------------------------------------------------------------*
   * @brief Template specialization for operator() for types whose values can be
   * convertible to a 0 or 1 to represent false/true. The converting is done by
   * checking against the default and user-specified true/false values list.
   *
   * It is handled here rather than within convertStrToValue() as that function
   * is used by other types (ex. timestamp) that aren't 'booleable'.
   *---------------------------------------------------------------------------**/
  template <typename T,
            typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ void operator()(
      const char *csvData, void *gdfColumnData, long rowIndex, long start,
      long end, const ParseOptions &opts) {
    T &value{static_cast<T *>(gdfColumnData)[rowIndex]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - start + 1;
    if (serializedTrieContains(opts.trueValuesTrie, csvData + start, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, csvData + start, field_len)) {
      value = 0;
    } else {
      value = convertStrToValue<T>(csvData, start, end, opts);
    }
  }

  /**---------------------------------------------------------------------------*
   * @brief Default template operator() dispatch specialization all data types
   * (including wrapper types) that is not covered by above.
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
 * @param[out] num_valid The numbers of valid fields in columns
 *
 * @return gdf_error GDF_SUCCESS upon completion
 *---------------------------------------------------------------------------**/
__global__ void convertCsvToGdf(char *raw_csv, const ParseOptions opts,
                                gdf_size_type num_records, int num_columns,
                                bool *parseCol, uint64_t *recStart,
                                gdf_dtype *dtype, void **gdf_data,
                                gdf_valid_type **valid,
                                gdf_size_type *num_valid)
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

	while(col<num_columns){

		if(start>stop)
			break;

		pos = seekFieldEnd(raw_csv, opts, pos, stop);

		if(parseCol[col]==true){

			// check if the entire field is a NaN string - consistent with pandas
			const bool is_na = serializedTrieContains(opts.naValuesTrie, raw_csv + start, pos - start);

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
					auto str_list = static_cast<string_pair*>(gdf_data[actual_col]);
					str_list[rec_id].first = raw_csv + start;
					str_list[rec_id].second = end - start;
				} else {
					cudf::type_dispatcher(
						dtype[actual_col], ConvertFunctor{}, raw_csv,
						gdf_data[actual_col], rec_id, start, tempPos, opts);
				}

				// set the valid bitmap - all bits were set to 0 to start
				setBitmapBit(valid[actual_col], rec_id);
				atomicAdd(&num_valid[actual_col], 1);
			}
			else if(dtype[actual_col]==gdf_dtype::GDF_STRING){
				auto str_list = static_cast<string_pair*>(gdf_data[actual_col]);
				str_list[rec_id].first = nullptr;
				str_list[rec_id].second = 0;
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
      raw_csv->data.data(), raw_csv->opts, raw_csv->num_records,
      raw_csv->num_actual_cols, raw_csv->d_parseCol.data().get(), raw_csv->recStart.data(),
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
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] num_records The number of lines/rows of CSV data
 * @param[in] num_columns The number of columns of CSV data
 * @param[in] parseCol Whether to parse or skip a column
 * @param[in] recStart The start the CSV data of interest
 * @param[out] d_columnData The count for each column data type
 *
 * @returns GDF_SUCCESS upon successful computation
 *---------------------------------------------------------------------------**/
__global__
void dataTypeDetection(char *raw_csv,
                       const ParseOptions opts,
                       gdf_size_type num_records,
                       int num_columns,
                       bool *parseCol,
                       uint64_t *recStart,
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
			long countExponent=0;

			// Modify start & end to ignore whitespace and quotechars
			// This could possibly result in additional empty fields
			adjustForWhitespaceAndQuotes(raw_csv, &start, &tempPos);

			const long strLen = tempPos - start + 1;

			const bool maybe_hex = ((strLen > 2 && raw_csv[start] == '0' && raw_csv[start + 1] == 'x') ||
				(strLen > 3 && raw_csv[start] == '-' && raw_csv[start + 1] == '0' && raw_csv[start + 2] == 'x'));

			for(long startPos=start; startPos<=tempPos; startPos++){
				if(isDigit(raw_csv[startPos], maybe_hex)){
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
					case 'e':
					case 'E':
						if (!maybe_hex && startPos > start && startPos < tempPos) 
							countExponent++;break;
					default:
						countString++;
						break;	
				}
			}

			// Integers have to have the length of the string
			long int_req_number_cnt = strLen;
			// Off by one if they start with a minus sign
			if(raw_csv[start]=='-' && strLen > 1){
				--int_req_number_cnt;
			}
			// Off by one if they are a hexadecimal number
			if(maybe_hex) {
				--int_req_number_cnt;
			}

			if(strLen==0){ // Removed spaces ' ' in the pre-processing and thus we can have an empty string.
				atomicAdd(& d_columnData[actual_col].countNULL, 1L);
			}
			else if(countNumber==int_req_number_cnt){
				// Checking to see if we the integer value requires 8,16,32,64 bits.
				// This will allow us to allocate the exact amount of memory.
				const auto value = convertStrToValue<int64_t>(raw_csv, start, tempPos, opts);
				const size_t field_len = tempPos - start + 1;
				if (serializedTrieContains(opts.trueValuesTrie, raw_csv + start, field_len) ||
					serializedTrieContains(opts.falseValuesTrie, raw_csv + start, field_len)){
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
			else if(isLikeFloat(strLen, countNumber, countDecimal, countDash, countExponent)){
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
