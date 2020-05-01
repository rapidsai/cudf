/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <io/utilities/legacy/parsing_utils.cuh>
#include "csv_gpu.h"

namespace cudf {
namespace io {
namespace csv {
namespace gpu {
/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] num_records The number of lines/rows of CSV data
 * @param[in] num_columns The number of columns of CSV data
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] recStart The start the CSV data of interest
 * @param[out] d_columnData The count for each column data type
 **/
__global__ void dataTypeDetection(const char *raw_csv,
                                  const ParseOptions opts,
                                  cudf::size_type num_records,
                                  int num_columns,
                                  column_parse::flags *flags,
                                  const uint64_t *recStart,
                                  column_parse::stats *d_columnData)
{
  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  long rec_id = threadIdx.x + (blockDim.x * blockIdx.x);

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id >= num_records) { return; }

  long start = recStart[rec_id];
  long stop  = recStart[rec_id + 1];

  long pos       = start;
  int col        = 0;
  int actual_col = 0;

  // Going through all the columns of a given record
  while (col < num_columns) {
    if (start > stop) { break; }

    pos = seekFieldEnd(raw_csv, opts, pos, stop);

    // Checking if this is a column that the user wants --- user can filter
    // columns
    if (flags[col] & column_parse::enabled) {
      long tempPos   = pos - 1;
      long field_len = pos - start;

      if (field_len <= 0 || serializedTrieContains(opts.naValuesTrie, raw_csv + start, field_len)) {
        atomicAdd(&d_columnData[actual_col].countNULL, 1);
      } else if (serializedTrieContains(opts.trueValuesTrie, raw_csv + start, field_len) ||
                 serializedTrieContains(opts.falseValuesTrie, raw_csv + start, field_len)) {
        atomicAdd(&d_columnData[actual_col].countBool, 1);
      } else {
        long countNumber   = 0;
        long countDecimal  = 0;
        long countSlash    = 0;
        long countDash     = 0;
        long countPlus     = 0;
        long countColon    = 0;
        long countString   = 0;
        long countExponent = 0;

        // Modify start & end to ignore whitespace and quotechars
        // This could possibly result in additional empty fields
        adjustForWhitespaceAndQuotes(raw_csv, &start, &tempPos);
        field_len = tempPos - start + 1;

        for (long startPos = start; startPos <= tempPos; startPos++) {
          if (isDigit(raw_csv[startPos])) {
            countNumber++;
            continue;
          }
          // Looking for unique characters that will help identify column types.
          switch (raw_csv[startPos]) {
            case '.': countDecimal++; break;
            case '-': countDash++; break;
            case '+': countPlus++; break;
            case '/': countSlash++; break;
            case ':': countColon++; break;
            case 'e':
            case 'E':
              if (startPos > start && startPos < tempPos) countExponent++;
              break;
            default: countString++; break;
          }
        }

        // Integers have to have the length of the string
        long int_req_number_cnt = field_len;
        // Off by one if they start with a minus sign
        if ((raw_csv[start] == '-' || raw_csv[start] == '+') && field_len > 1) {
          --int_req_number_cnt;
        }

        if (field_len == 0) {
          // Ignoring whitespace and quotes can result in empty fields
          atomicAdd(&d_columnData[actual_col].countNULL, 1);
        } else if (flags[col] & column_parse::as_datetime) {
          // PANDAS uses `object` dtype if the date is unparseable
          if (isLikeDateTime(countString, countDecimal, countColon, countDash, countSlash)) {
            atomicAdd(&d_columnData[actual_col].countDateAndTime, 1);
          } else {
            atomicAdd(&d_columnData[actual_col].countString, 1);
          }
        } else if (countNumber == int_req_number_cnt) {
          // Checking to see if we the integer value requires 8,16,32,64 bits.
          // This will allow us to allocate the exact amount of memory.
          const auto value = convertStrToValue<int64_t>(raw_csv, start, tempPos, opts);
          if (value >= (1L << 31)) {
            atomicAdd(&d_columnData[actual_col].countInt64, 1);
          } else if (value >= (1L << 15)) {
            atomicAdd(&d_columnData[actual_col].countInt32, 1);
          } else if (value >= (1L << 7)) {
            atomicAdd(&d_columnData[actual_col].countInt16, 1);
          } else {
            atomicAdd(&d_columnData[actual_col].countInt8, 1);
          }
        } else if (isLikeFloat(
                     field_len, countNumber, countDecimal, countDash + countPlus, countExponent)) {
          atomicAdd(&d_columnData[actual_col].countFloat, 1);
        } else {
          atomicAdd(&d_columnData[actual_col].countString, 1);
        }
      }
      actual_col++;
    }
    pos++;
    start = pos;
    col++;
  }
}

/**
 * @brief Functor for converting CSV data to cuDF data type value.
 **/
struct ConvertFunctor {
  /**
   * @brief Template specialization for operator() for types whose values can be
   * convertible to a 0 or 1 to represent false/true. The converting is done by
   * checking against the default and user-specified true/false values list.
   *
   * It is handled here rather than within convertStrToValue() as that function
   * is used by other types (ex. timestamp) that aren't 'booleable'.
   **/
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ void operator()(const char *csvData,
                                                      void *gdfColumnData,
                                                      long rowIndex,
                                                      long start,
                                                      long end,
                                                      const ParseOptions &opts,
                                                      column_parse::flags flags)
  {
    T &value{static_cast<T *>(gdfColumnData)[rowIndex]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - start + 1;
    if (serializedTrieContains(opts.trueValuesTrie, csvData + start, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, csvData + start, field_len)) {
      value = 0;
    } else {
      if (flags & column_parse::as_hexadecimal) {
        value = convertStrToValue<T, 16>(csvData, start, end, opts);
      } else {
        value = convertStrToValue<T>(csvData, start, end, opts);
      }
    }
  }

  /**
   * @brief Default template operator() dispatch specialization all data types
   * (including wrapper types) that is not covered by above.
   **/
  template <typename T, typename std::enable_if_t<!std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ void operator()(const char *csvData,
                                                      void *gdfColumnData,
                                                      long rowIndex,
                                                      long start,
                                                      long end,
                                                      const ParseOptions &opts,
                                                      column_parse::flags flags)
  {
    T &value{static_cast<T *>(gdfColumnData)[rowIndex]};
    value = convertStrToValue<T>(csvData, start, end, opts);
  }
};

/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] num_records The number of lines/rows of CSV data
 * @param[in] num_columns The number of columns of CSV data
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] recStart The start the CSV data of interest
 * @param[in] dtype The data type of the column
 * @param[out] data The output column data
 * @param[out] valid The bitmaps indicating whether column fields are valid
 * @param[out] num_valid The numbers of valid fields in columns
 **/
__global__ void convertCsvToGdf(const char *raw_csv,
                                const ParseOptions opts,
                                cudf::size_type num_records,
                                int num_columns,
                                const column_parse::flags *flags,
                                const uint64_t *recStart,
                                gdf_dtype *dtype,
                                void **data,
                                cudf::valid_type **valid,
                                cudf::size_type *num_valid)
{
  // thread IDs range per block, so also need the block id
  long rec_id =
    threadIdx.x + (blockDim.x * blockIdx.x);  // this is entry into the field array - tid is
                                              // an elements within the num_entries array

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id >= num_records) return;

  long start = recStart[rec_id];
  long stop  = recStart[rec_id + 1];

  long pos       = start;
  int col        = 0;
  int actual_col = 0;

  while (col < num_columns) {
    if (start > stop) break;

    pos = seekFieldEnd(raw_csv, opts, pos, stop);

    if (flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      const bool is_na = serializedTrieContains(opts.naValuesTrie, raw_csv + start, pos - start);

      // Modify start & end to ignore whitespace and quotechars
      long tempPos = pos - 1;
      if (!is_na && dtype[actual_col] != gdf_dtype::GDF_CATEGORY &&
          dtype[actual_col] != gdf_dtype::GDF_STRING) {
        adjustForWhitespaceAndQuotes(raw_csv, &start, &tempPos, opts.quotechar);
      }

      if (!is_na && start <= (tempPos)) {  // Empty fields are not legal values

        // Type dispatcher does not handle GDF_STRINGS
        if (dtype[actual_col] == gdf_dtype::GDF_STRING) {
          long end = pos;
          if (opts.keepquotes == false) {
            if ((raw_csv[start] == opts.quotechar) && (raw_csv[end - 1] == opts.quotechar)) {
              start++;
              end--;
            }
          }
          auto str_list          = static_cast<std::pair<const char *, size_t> *>(data[actual_col]);
          str_list[rec_id].first = raw_csv + start;
          str_list[rec_id].second = end - start;
        } else {
          cudf::type_dispatcher(dtype[actual_col],
                                ConvertFunctor{},
                                raw_csv,
                                data[actual_col],
                                rec_id,
                                start,
                                tempPos,
                                opts,
                                flags[col]);
        }

        // set the valid bitmap - all bits were set to 0 to start
        setBitmapBit(valid[actual_col], rec_id);
        atomicAdd(&num_valid[actual_col], 1);
      } else if (dtype[actual_col] == gdf_dtype::GDF_STRING) {
        auto str_list           = static_cast<std::pair<const char *, size_t> *>(data[actual_col]);
        str_list[rec_id].first  = nullptr;
        str_list[rec_id].second = 0;
      }
      actual_col++;
    }
    pos++;
    start = pos;
    col++;
  }
}

cudaError_t __host__ DetectCsvDataTypes(const char *data,
                                        const uint64_t *row_starts,
                                        cudf::size_type num_rows,
                                        cudf::size_type num_columns,
                                        const ParseOptions &options,
                                        column_parse::flags *flags,
                                        column_parse::stats *stats,
                                        cudaStream_t stream)
{
  int blockSize;    // suggested thread count to use
  int minGridSize;  // minimum block count required
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dataTypeDetection));

  // Calculate actual block count to use based on records count
  const int gridSize = (num_rows + blockSize - 1) / blockSize;

  dataTypeDetection<<<gridSize, blockSize, 0, stream>>>(
    data, options, num_rows, num_columns, flags, row_starts, stats);

  return cudaSuccess;
}

cudaError_t __host__ DecodeCsvColumnData(const char *data,
                                         const uint64_t *row_starts,
                                         cudf::size_type num_rows,
                                         cudf::size_type num_columns,
                                         const ParseOptions &options,
                                         const column_parse::flags *flags,
                                         gdf_dtype *dtypes,
                                         void **columns,
                                         cudf::valid_type **valids,
                                         cudf::size_type *num_valid,
                                         cudaStream_t stream)
{
  int blockSize;    // suggested thread count to use
  int minGridSize;  // minimum block count required
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, convertCsvToGdf));

  // Calculate actual block count to use based on records count
  const int gridSize = (num_rows + blockSize - 1) / blockSize;

  convertCsvToGdf<<<gridSize, blockSize, 0, stream>>>(
    data, options, num_rows, num_columns, flags, row_starts, dtypes, columns, valids, num_valid);

  return cudaSuccess;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
