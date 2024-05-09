/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#pragma once

#include "error.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/parquet/parquet.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/statistics/statistics.cuh"
#include "io/utilities/column_buffer.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <type_traits>
#include <vector>

namespace cudf::io::parquet::detail {

using cudf::io::detail::string_index_pair;

// Largest number of bits to use for dictionary keys
constexpr int MAX_DICT_BITS = 24;

// Total number of unsigned 24 bit values
constexpr size_type MAX_DICT_SIZE = (1 << MAX_DICT_BITS) - 1;

// level decode buffer size.
constexpr int LEVEL_DECODE_BUF_SIZE = 2048;

template <int rolling_size>
constexpr int rolling_index(int index)
{
  // Cannot divide by 0. But `rolling_size` will be 0 for unused arrays, so this case will never
  // actual be executed.
  if constexpr (rolling_size == 0) {
    return index;
  } else {
    return index % rolling_size;
  }
}

// PARQUET-2261 allows for not writing the level histograms in certain cases.
// The repetition level histogram may be omitted when max_rep_level equals 0. The definition
// level histogram may be omitted when max_def_level equals 0 or 1. In the case of
// max_rep_level == 0, the rep histogram would have a single value equal to num_rows. In the
// case of max_def_level == 0, the def histogram would have a single value equal to num_values,
// and when max_def_level == 1, the histogram would be {num_nulls, num_values - num_nulls}.
//
// These constants control libcudf's behavior. Currently, each histogram will be written when
// max level is greater than 0. Even though this leads to some redundancy in the max_def_level == 1
// case, having the histogram data relieves the reader from having to reconstruct it from the
// OffsetIndex and ColumnMetaData.
constexpr uint8_t REP_LVL_HIST_CUTOFF = 0;
constexpr uint8_t DEF_LVL_HIST_CUTOFF = 0;

// see setupLocalPageInfo() in page_decode.cuh for supported page encodings
constexpr bool is_supported_encoding(Encoding enc)
{
  switch (enc) {
    case Encoding::PLAIN:
    case Encoding::PLAIN_DICTIONARY:
    case Encoding::RLE:
    case Encoding::RLE_DICTIONARY:
    case Encoding::DELTA_BINARY_PACKED:
    case Encoding::DELTA_LENGTH_BYTE_ARRAY:
    case Encoding::DELTA_BYTE_ARRAY:
    case Encoding::BYTE_STREAM_SPLIT: return true;
    default: return false;
  }
}

/**
 * @brief Atomically OR `error` into `error_code`.
 */
constexpr void set_error(kernel_error::value_type error, kernel_error::pointer error_code)
{
  if (error != 0) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_device> ref{*error_code};
    ref.fetch_or(error, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Enum for the different types of errors that can occur during decoding.
 *
 * These values are used as bitmasks, so they must be powers of 2.
 */
enum class decode_error : kernel_error::value_type {
  DATA_STREAM_OVERRUN      = 0x1,
  LEVEL_STREAM_OVERRUN     = 0x2,
  UNSUPPORTED_ENCODING     = 0x4,
  INVALID_LEVEL_RUN        = 0x8,
  INVALID_DATA_TYPE        = 0x10,
  EMPTY_PAGE               = 0x20,
  INVALID_DICT_WIDTH       = 0x40,
  DELTA_PARAM_MISMATCH     = 0x80,
  DELTA_PARAMS_UNSUPPORTED = 0x100,
};

/**
 * @brief Struct representing an input column in the file.
 */
struct input_column_info {
  int schema_idx;
  std::string name;
  bool has_repetition;
  // size == nesting depth. the associated real output
  // buffer index in the dest column for each level of nesting.
  std::vector<int> nesting;

  input_column_info(int _schema_idx, std::string _name, bool _has_repetition)
    : schema_idx(_schema_idx), name(_name), has_repetition(_has_repetition)
  {
  }

  auto nesting_depth() const { return nesting.size(); }
};

// The delta encodings use ULEB128 integers, but parquet only uses max 64 bits.
using uleb128_t   = uint64_t;
using zigzag128_t = int64_t;

// this is in C++23
#if !defined(__cpp_lib_is_scoped_enum)
template <typename Enum, bool = std::is_enum_v<Enum>>
struct is_scoped_enum {
  static const bool value = not std::is_convertible_v<Enum, std::underlying_type_t<Enum>>;
};

template <typename Enum>
struct is_scoped_enum<Enum, false> {
  static const bool value = false;
};
#else
using std::is_scoped_enum;
#endif

// helpers to do bit operations on scoped enums
template <class T1,
          class T2,
          typename std::enable_if_t<(is_scoped_enum<T1>::value and std::is_same_v<T1, T2>) or
                                    (is_scoped_enum<T1>::value and std::is_same_v<uint32_t, T2>) or
                                    (is_scoped_enum<T2>::value and std::is_same_v<uint32_t, T1>)>* =
            nullptr>
constexpr uint32_t BitAnd(T1 a, T2 b)
{
  return static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
}

template <class T1,
          class T2,
          typename std::enable_if_t<(is_scoped_enum<T1>::value and std::is_same_v<T1, T2>) or
                                    (is_scoped_enum<T1>::value and std::is_same_v<uint32_t, T2>) or
                                    (is_scoped_enum<T2>::value and std::is_same_v<uint32_t, T1>)>* =
            nullptr>
constexpr uint32_t BitOr(T1 a, T2 b)
{
  return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

/**
 * @brief Enums for the flags in the page header
 */
enum {
  PAGEINFO_FLAGS_DICTIONARY = (1 << 0),  // Indicates a dictionary page
  PAGEINFO_FLAGS_V2         = (1 << 1),  // V2 page header
};

/**
 * @brief Enum for the two encoding streams
 */
enum level_type {
  DEFINITION = 0,
  REPETITION,

  NUM_LEVEL_TYPES
};

/**
 * @brief Enum of mask bits for the PageInfo kernel_mask
 *
 * Used to control which decode kernels to run.
 */
enum class decode_kernel_mask {
  NONE                   = 0,
  GENERAL                = (1 << 0),  // Run catch-all decode kernel
  STRING                 = (1 << 1),  // Run decode kernel for string data
  DELTA_BINARY           = (1 << 2),  // Run decode kernel for DELTA_BINARY_PACKED data
  DELTA_BYTE_ARRAY       = (1 << 3),  // Run decode kernel for DELTA_BYTE_ARRAY encoded data
  DELTA_LENGTH_BA        = (1 << 4),  // Run decode kernel for DELTA_LENGTH_BYTE_ARRAY encoded data
  FIXED_WIDTH_NO_DICT    = (1 << 5),  // Run decode kernel for fixed width non-dictionary pages
  FIXED_WIDTH_DICT       = (1 << 6),  // Run decode kernel for fixed width dictionary pages
  BYTE_STREAM_SPLIT      = (1 << 7),  // Run decode kernel for BYTE_STREAM_SPLIT encoded data
  BYTE_STREAM_SPLIT_FLAT = (1 << 8),  // Same as above but with a flat schema
};

// mask representing all the ways in which a string can be encoded
constexpr uint32_t STRINGS_MASK =
  BitOr(BitOr(decode_kernel_mask::DELTA_BYTE_ARRAY, decode_kernel_mask::STRING),
        decode_kernel_mask::DELTA_LENGTH_BA);
/**
 * @brief Nesting information specifically needed by the decode and preprocessing
 * kernels.
 *
 * This data is kept separate from PageNestingInfo to keep it as small as possible.
 * It is used in a cached form in shared memory when possible.
 */
struct PageNestingDecodeInfo {
  // set up prior to decoding
  int32_t max_def_level;
  // input repetition/definition levels are remapped with these values
  // into the corresponding real output nesting depths.
  int32_t start_depth;
  int32_t end_depth;

  // computed during preprocessing
  int32_t page_start_value;

  // computed during decoding
  int32_t null_count;

  // used internally during decoding
  int32_t valid_map_offset;
  int32_t valid_count;
  int32_t value_count;
  uint8_t* data_out;
  uint8_t* string_out;
  bitmask_type* valid_map;
};

// Use up to 512 bytes of shared memory as a cache for nesting information.
// As of 1/20/23, this gives us a max nesting depth of 10 (after which it falls back to
// global memory). This handles all but the most extreme cases.
constexpr int max_cacheable_nesting_decode_info = (512) / sizeof(PageNestingDecodeInfo);

/**
 * @brief Nesting information
 *
 * This struct serves two purposes:
 *
 * - It stores information about output (cudf) columns
 * - It provides a mapping from input column depth to output column depth via
 * the start_depth and end_depth fields.
 *
 */
struct PageNestingInfo {
  // set at initialization (see start_offset_output_iterator in reader_impl_preprocess.cu)
  cudf::type_id type;  // type of the corresponding cudf output column
  bool nullable;

  // TODO: these fields might make sense to move into PageNestingDecodeInfo for memory performance
  // reasons.
  int32_t size;  // this page/nesting-level's row count contribution to the output column, if fully
                 // decoded
  int32_t batch_size;  // the size of the page for this batch
};

/**
 * @brief Struct describing a particular page of column chunk data
 */
struct PageInfo {
  uint8_t* page_data;  // Compressed page data before decompression, or uncompressed data after
                       // decompression
  int32_t compressed_page_size;    // compressed data size in bytes
  int32_t uncompressed_page_size;  // uncompressed data size in bytes
  // for V2 pages, the def and rep level data is not compressed, and lacks the 4-byte length
  // indicator. instead the lengths for these are stored in the header.
  int32_t lvl_bytes[level_type::NUM_LEVEL_TYPES];  // length of the rep/def levels (V2 header)
  // Number of values in this data page or dictionary.
  // Important : the # of input values does not necessarily
  // correspond to the number of rows in the output. It just reflects the number
  // of values in the input stream.
  // - In the case of a flat schema, it will correspond to the # of output rows
  // - In the case of a nested schema, you have to decode the repetition and definition
  //   levels to extract actual column values
  int32_t num_input_values;
  int32_t chunk_row;  // starting row of this page relative to the start of the chunk
  int32_t num_rows;   // number of rows in this page
  // the next four are calculated in gpuComputePageStringSizes
  int32_t num_nulls;       // number of null values (V2 header), but recalculated for string cols
  int32_t num_valids;      // number of non-null values, taking into account skip_rows/num_rows
  int32_t start_val;       // index of first value of the string data stream to use
  int32_t end_val;         // index of last value in string data stream
  int32_t chunk_idx;       // column chunk this page belongs to
  int32_t src_col_schema;  // schema index of this column
  uint8_t flags;           // PAGEINFO_FLAGS_XXX
  Encoding encoding;       // Encoding for data or dictionary page
  Encoding definition_level_encoding;  // Encoding used for definition levels (data page)
  Encoding repetition_level_encoding;  // Encoding used for repetition levels (data page)

  // for nested types, we run a preprocess step in order to determine output
  // column sizes. Because of this, we can jump directly to the position in the
  // input data to start decoding instead of reading all of the data and discarding
  // rows we don't care about.
  //
  // NOTE: for flat hierarchies we do not do the preprocess step, so skipped_values and
  // skipped_leaf_values will always be 0.
  //
  // # of values skipped in the repetition/definition level stream
  int32_t skipped_values;
  // # of values skipped in the actual data stream.
  int32_t skipped_leaf_values;
  // for string columns only, the size of all the chars in the string for
  // this page. only valid/computed during the base preprocess pass
  size_t str_offset;  // offset into string data for this page
  int32_t str_bytes;
  bool has_page_index;  // true if str_bytes, num_valids, etc are derivable from page indexes

  // nesting information (input/output) for each page. this array contains
  // input column nesting information, output column nesting information and
  // mappings between the two. the length of the array, nesting_info_size is
  // max(num_output_nesting_levels, max_definition_levels + 1)
  int32_t num_output_nesting_levels;
  int32_t nesting_info_size;
  PageNestingInfo* nesting;
  PageNestingDecodeInfo* nesting_decode;

  // level decode buffers
  uint8_t* lvl_decode_buf[level_type::NUM_LEVEL_TYPES];

  // temporary space for decoding DELTA_BYTE_ARRAY encoded strings
  int64_t temp_string_size;
  uint8_t* temp_string_buf;

  decode_kernel_mask kernel_mask;

  // str_bytes from page index. because str_bytes needs to be reset each iteration
  // while doing chunked reads, persist the value from the page index here.
  int32_t str_bytes_from_index;
};

// forward declaration
struct column_chunk_info;

/**
 * @brief Return the column schema id as the key for a PageInfo struct.
 */
struct get_page_key {
  __device__ int32_t operator()(PageInfo const& page) const { return page.src_col_schema; }
};

/**
 * @brief Return an iterator that returns they keys for a vector of pages.
 */
inline auto make_page_key_iterator(device_span<PageInfo const> pages)
{
  return thrust::make_transform_iterator(pages.begin(), get_page_key{});
}

/**
 * @brief Struct describing a particular chunk of column data
 */
struct ColumnChunkDesc {
  constexpr ColumnChunkDesc() noexcept {};
  explicit ColumnChunkDesc(size_t compressed_size_,
                           uint8_t* compressed_data_,
                           size_t num_values_,
                           Type datatype_,
                           int32_t datatype_length_,
                           size_t start_row_,
                           uint32_t num_rows_,
                           int16_t max_definition_level_,
                           int16_t max_repetition_level_,
                           int16_t max_nesting_depth_,
                           uint8_t def_level_bits_,
                           uint8_t rep_level_bits_,
                           Compression codec_,
                           thrust::optional<LogicalType> logical_type_,
                           int32_t ts_clock_rate_,
                           int32_t src_col_index_,
                           int32_t src_col_schema_,
                           column_chunk_info const* chunk_info_,
                           float list_bytes_per_row_est_,
                           bool strings_to_categorical_)
    : compressed_data(compressed_data_),
      compressed_size(compressed_size_),
      num_values(num_values_),
      start_row(start_row_),
      num_rows(num_rows_),
      max_level{max_definition_level_, max_repetition_level_},
      max_nesting_depth{max_nesting_depth_},
      type_length(datatype_length_),
      physical_type(datatype_),
      level_bits{def_level_bits_, rep_level_bits_},
      num_data_pages(0),
      num_dict_pages(0),
      dict_page(nullptr),
      str_dict_index(nullptr),
      valid_map_base{nullptr},
      column_data_base{nullptr},
      column_string_base{nullptr},
      codec(codec_),
      logical_type(logical_type_),
      ts_clock_rate(ts_clock_rate_),
      src_col_index(src_col_index_),
      src_col_schema(src_col_schema_),
      h_chunk_info(chunk_info_),
      list_bytes_per_row_est(list_bytes_per_row_est_),
      is_strings_to_cat(strings_to_categorical_),
      is_large_string_col(false)
  {
  }

  uint8_t const* compressed_data{};  // pointer to compressed column chunk data
  size_t compressed_size{};          // total compressed data size for this chunk
  size_t num_values{};               // total number of values in this column
  size_t start_row{};                // file-wide, absolute starting row of this chunk
  uint32_t num_rows{};               // number of rows in this chunk
  int16_t max_level[level_type::NUM_LEVEL_TYPES]{};  // max definition/repetition level
  int16_t max_nesting_depth{};                       // max nesting depth of the output
  int32_t type_length{};                             // type length from schema (for FLBA only)
  Type physical_type{};                              // parquet physical data type
  uint8_t
    level_bits[level_type::NUM_LEVEL_TYPES]{};  // bits to encode max definition/repetition levels
  int32_t num_data_pages{};                     // number of data pages
  int32_t num_dict_pages{};                     // number of dictionary pages
  PageInfo const* dict_page{};
  string_index_pair* str_dict_index{};           // index for string dictionary
  bitmask_type** valid_map_base{};               // base pointers of valid bit map for this column
  void** column_data_base{};                     // base pointers of column data
  void** column_string_base{};                   // base pointers of column string data
  Compression codec{};                           // compressed codec enum
  thrust::optional<LogicalType> logical_type{};  // logical type
  int32_t ts_clock_rate{};  // output timestamp clock frequency (0=default, 1000=ms, 1000000000=ns)

  int32_t src_col_index{};   // my input column index
  int32_t src_col_schema{};  // my schema index in the file

  // pointer to column_chunk_info struct for this chunk (host only)
  column_chunk_info const* h_chunk_info{};

  float list_bytes_per_row_est{};  // for LIST columns, an estimate on number of bytes per row

  bool is_strings_to_cat{};    // convert strings to hashes
  bool is_large_string_col{};  // `true` if string data uses 64-bit offsets
};

/**
 * @brief A utility structure for use in decoding page headers.
 */
struct chunk_page_info {
  PageInfo* pages;
};

/**
 * @brief Struct describing an encoder column
 */
struct parquet_column_device_view : stats_column_desc {
  Type physical_type;            //!< physical data type
  ConvertedType converted_type;  //!< logical data type
  int32_t type_length;           //!< length of fixed_length_byte_array data
  uint8_t level_bits;  //!< bits to encode max definition (lower nibble) & repetition (upper nibble)
                       //!< levels
  constexpr uint8_t num_def_level_bits() const { return level_bits & 0xf; }
  constexpr uint8_t num_rep_level_bits() const { return level_bits >> 4; }
  uint8_t max_def_level;  //!< needed for SizeStatistics calculation
  uint8_t max_rep_level;

  size_type const* level_offsets;  //!< Offset array for per-row pre-calculated rep/def level values
  uint8_t const* rep_values;       //!< Pre-calculated repetition level values
  uint8_t const* def_values;       //!< Pre-calculated definition level values
  uint8_t const* nullability;  //!< Array of nullability of each nesting level. e.g. nullable[0] is
                               //!< nullability of parent_column. May be different from
                               //!< col.nullable() in case of chunked writing.
  bool output_as_byte_array;   //!< Indicates this list column is being written as a byte array
  bool skip_compression;       //!< Skip compression for this column
  column_encoding requested_encoding;  //!< User specified encoding for this column.
};

struct EncColumnChunk;

/**
 * @brief Struct describing an encoder page fragment
 */
struct PageFragment {
  uint32_t fragment_data_size;  //!< Size of fragment data in bytes
  uint32_t dict_data_size;      //!< Size of dictionary for this fragment
  uint32_t num_values;  //!< Number of values in fragment. Different from num_rows for nested type
  uint32_t start_value_idx;
  uint32_t num_leaf_values;  //!< Number of leaf values in fragment. Does not include nulls at
                             //!< non-leaf level
  uint32_t num_valid;        //<! Number of non-null leaf values
  size_type start_row;       //!< First row in fragment
  uint16_t num_rows;         //!< Number of rows in fragment
  uint16_t num_dict_vals;    //!< Number of unique dictionary entries
  EncColumnChunk* chunk;     //!< The chunk that this fragment belongs to
};

/// Size of hash used for building dictionaries
constexpr unsigned int kDictHashBits = 16;
constexpr size_t kDictScratchSize    = (1 << kDictHashBits) * sizeof(uint32_t);

struct EncPage;
struct slot_type;

// convert Encoding to a mask value
constexpr uint32_t encoding_to_mask(Encoding encoding)
{
  return 1 << static_cast<uint32_t>(encoding);
}

/**
 * @brief Enum of mask bits for the EncPage kernel_mask
 *
 * Used to control which encode kernels to run.
 */
enum class encode_kernel_mask {
  PLAIN             = (1 << 0),  // Run plain encoding kernel
  DICTIONARY        = (1 << 1),  // Run dictionary encoding kernel
  DELTA_BINARY      = (1 << 2),  // Run DELTA_BINARY_PACKED encoding kernel
  DELTA_LENGTH_BA   = (1 << 3),  // Run DELTA_LENGTH_BYTE_ARRAY encoding kernel
  DELTA_BYTE_ARRAY  = (1 << 4),  // Run DELTA_BYtE_ARRAY encoding kernel
  BYTE_STREAM_SPLIT = (1 << 5),  // Run plain encoding kernel, but split streams
};

/**
 * @brief Struct describing an encoder column chunk
 */
struct EncColumnChunk {
  parquet_column_device_view const* col_desc;  //!< Column description
  size_type col_desc_id;
  PageFragment* fragments;        //!< First fragment in chunk
  uint8_t* uncompressed_bfr;      //!< Uncompressed page data
  uint8_t* compressed_bfr;        //!< Compressed page data
  statistics_chunk const* stats;  //!< Fragment statistics
  uint32_t bfr_size;              //!< Uncompressed buffer size
  uint32_t compressed_size;       //!< Compressed buffer size
  uint32_t max_page_data_size;    //!< Max data size (excluding header) of any page in this chunk
  uint32_t page_headers_size;     //!< Sum of size of all page headers
  size_type start_row;            //!< First row of chunk
  uint32_t num_rows;              //!< Number of rows in chunk
  size_type num_values;     //!< Number of values in chunk. Different from num_rows for nested types
  uint32_t first_fragment;  //!< First fragment of chunk
  EncPage* pages;           //!< Ptr to pages that belong to this chunk
  uint32_t first_page;      //!< First page of chunk
  uint32_t num_pages;       //!< Number of pages in chunk
  uint8_t is_compressed;    //!< Nonzero if the chunk uses compression
  uint32_t dictionary_size;    //!< Size of dictionary page including header
  uint32_t ck_stat_size;       //!< Size of chunk-level statistics (included in 1st page header)
  slot_type* dict_map_slots;   //!< Hash map storage for calculating dict encoding for this chunk
  size_type dict_map_size;     //!< Size of dict_map_slots
  size_type num_dict_entries;  //!< Total number of entries in dictionary
  size_type
    uniq_data_size;  //!< Size of dictionary page (set of all unique values) if dict enc is used
  size_type plain_data_size;  //!< Size of data in this chunk if plain encoding is used
  size_type* dict_data;       //!< Dictionary data (unique row indices)
  size_type* dict_index;  //!< Index of value in dictionary page. column[dict_data[dict_index[row]]]
  uint8_t dict_rle_bits;  //!< Bit size for encoding dictionary indices
  bool use_dictionary;    //!< True if the chunk uses dictionary encoding
  uint8_t* column_index_blob;    //!< Binary blob containing encoded column index for this chunk
  uint32_t column_index_size;    //!< Size of column index blob
  uint32_t encodings;            //!< Mask representing the set of encodings used for this chunk
  uint32_t* def_histogram_data;  //!< Buffers for size histograms. One for chunk and one per page.
  uint32_t* rep_histogram_data;  //!< Size is (max(level) + 1) * (num_data_pages + 1).
  size_t var_bytes_size;         //!< Sum of var_bytes_size from the pages (byte arrays only)

  constexpr uint32_t num_dict_pages() const { return use_dictionary ? 1 : 0; }

  constexpr uint32_t num_data_pages() const { return num_pages - num_dict_pages(); }
};

/**
 * @brief Struct describing an encoder data page
 */
struct EncPage {
  // all pointers at the top to keep things properly aligned
  uint8_t* page_data;            //!< Ptr to uncompressed page
  uint8_t* compressed_data;      //!< Ptr to compressed page
  EncColumnChunk* chunk;         //!< Chunk that this page belongs to
  compression_result* comp_res;  //!< Ptr to compression result
  uint32_t* def_histogram;       //!< Histogram of counts for each definition level
  uint32_t* rep_histogram;       //!< Histogram of counts for each repetition level
  // put this here in case it's ever made 64-bit
  encode_kernel_mask kernel_mask;  //!< Mask used to control which encoding kernels to run
  // the rest can be 4 byte aligned
  uint32_t chunk_id;         //!< Index in chunk array
  uint32_t hdr_size;         //!< Actual size of encoded page header
  uint32_t max_hdr_size;     //!< Maximum size of page header
  uint32_t max_data_size;    //!< Maximum size of encoded page data (excluding header)
  uint32_t data_size;        //!< Actual size of encoded page data (includes level data)
  uint32_t comp_data_size;   //!< Actual size of compressed page data
  uint32_t start_row;        //!< First row of page
  uint32_t num_rows;         //!< Rows in page
  uint32_t num_leaf_values;  //!< Values in page. Different from num_rows in case of nested types
  uint32_t num_values;  //!< Number of def/rep level values in page. Includes null/empty elements in
                        //!< non-leaf levels
  uint32_t def_lvl_bytes;   //!< Number of bytes of encoded definition level data
  uint32_t rep_lvl_bytes;   //!< Number of bytes of encoded repetition level data
  uint32_t max_lvl_size;    //!< Maximum size of level data (V2 only, 0 for V1)
  uint32_t num_nulls;       //!< Number of null values
  uint32_t num_valid;       //!< Number of valid leaf values
  uint32_t var_bytes_size;  //!< Number of variable length bytes in the page (byte arrays only)
  // enums and smaller stuff down here
  PageType page_type;      //!< Page type
  Encoding encoding;       //!< Encoding used for page data
  uint16_t num_fragments;  //!< Number of fragments in page

  constexpr bool is_v2() const { return page_type == PageType::DATA_PAGE_V2; }

  constexpr auto level_bytes() const { return def_lvl_bytes + rep_lvl_bytes; }
};

/**
 * @brief Test if the given column chunk is in a string column
 */
constexpr bool is_string_col(ColumnChunkDesc const& chunk)
{
  // return true for non-hashed byte_array and fixed_len_byte_array that isn't representing
  // a decimal.
  if (chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL) {
    return false;
  }

  auto const non_hashed_byte_array =
    chunk.physical_type == BYTE_ARRAY and not chunk.is_strings_to_cat;
  auto const fixed_len_byte_array = chunk.physical_type == FIXED_LEN_BYTE_ARRAY;
  return non_hashed_byte_array or fixed_len_byte_array;
}

/**
 * @brief Return true if the run with header run_header is a literal RLE run
 */
__device__ inline bool is_literal_run(int const run_header) { return (run_header & 1) == 1; }

/**
 * @brief Return true if the run with header run_header is a repeated RLE run
 */
__device__ inline bool is_repeated_run(int const run_header) { return !is_literal_run(run_header); }

/**
 * @brief Launches kernel for parsing the page headers in the column chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] chunk_pages List of pages associated with the chunks, in chunk-sorted order
 * @param[in] num_chunks Number of column chunks
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodePageHeaders(ColumnChunkDesc* chunks,
                       chunk_page_info* chunk_pages,
                       int32_t num_chunks,
                       kernel_error::pointer error_code,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for building the dictionary index for the column
 * chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use
 */
void BuildStringDictionaryIndex(ColumnChunkDesc* chunks,
                                int32_t num_chunks,
                                rmm::cuda_stream_view stream);

/**
 * @brief Get the set of kernels that need to be invoked on these pages as a bitmask.
 *
 * This function performs a bitwise OR on all of the individual `kernel_mask` fields on the pages
 * passed in.
 *
 * @param[in] pages List of pages to aggregate
 * @param[in] stream CUDA stream to use
 * @return Bitwise OR of all page `kernel_mask` values
 */
uint32_t GetAggregatedDecodeKernelMask(cudf::detail::hostdevice_span<PageInfo const> pages,
                                       rmm::cuda_stream_view stream);

/**
 * @brief Compute page output size information.
 *
 * When dealing with nested hierarchies (those that contain lists), or when doing a chunked
 * read, we need to obtain more information up front than we have with just the row counts.
 *
 * - We need to determine the sizes of each output cudf column per page
 * - We need to determine information about where to start decoding the value stream
 *   if we are using custom user bounds (skip_rows / num_rows)
 * - We need to determine actual number of top level rows per page
 * - If we are doing a chunked read, we need to determine the total string size per page
 *
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 * @param compute_num_rows If set to true, the num_rows field in PageInfo will be
 * computed
 * @param compute_string_sizes If set to true, the str_bytes field in PageInfo will
 * be computed
 * @param level_type_size Size in bytes of the type for level decoding
 * @param stream CUDA stream to use
 */
void ComputePageSizes(cudf::detail::hostdevice_span<PageInfo> pages,
                      cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                      size_t min_row,
                      size_t num_rows,
                      bool compute_num_rows,
                      bool compute_string_sizes,
                      int level_type_size,
                      rmm::cuda_stream_view stream);

/**
 * @brief Compute string page output size information.
 *
 * String columns need accurate data size information to preallocate memory in the column buffer to
 * store the char data. This calls a kernel to calculate information needed by the string decoding
 * kernel. On exit, the `str_bytes`, `num_nulls`, `num_valids`, and `str_offset` fields of the
 * PageInfo struct are updated. This call ignores non-string columns.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[out] temp_string_buf Temporary space needed for decoding DELTA_BYTE_ARRAY strings
 * @param[in] min_rows crop all rows below min_row
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[in] kernel_mask Mask of kernels to run
 * @param[in] stream CUDA stream to use
 */
void ComputePageStringSizes(cudf::detail::hostdevice_span<PageInfo> pages,
                            cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                            rmm::device_uvector<uint8_t>& temp_string_buf,
                            size_t min_row,
                            size_t num_rows,
                            int level_type_size,
                            uint32_t kernel_mask,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodePageData(cudf::detail::hostdevice_span<PageInfo> pages,
                    cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                    size_t num_rows,
                    size_t min_row,
                    int level_type_size,
                    kernel_error::pointer error_code,
                    rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the BYTE_STREAM_SPLIT column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeSplitPageData(cudf::detail::hostdevice_span<PageInfo> pages,
                         cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                         size_t num_rows,
                         size_t min_row,
                         int level_type_size,
                         kernel_error::pointer error_code,
                         rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the string column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeStringPageData(cudf::detail::hostdevice_span<PageInfo> pages,
                          cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                          size_t num_rows,
                          size_t min_row,
                          int level_type_size,
                          kernel_error::pointer error_code,
                          rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the DELTA_BINARY_PACKED column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeDeltaBinary(cudf::detail::hostdevice_span<PageInfo> pages,
                       cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                       size_t num_rows,
                       size_t min_row,
                       int level_type_size,
                       kernel_error::pointer error_code,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the DELTA_BYTE_ARRAY column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeDeltaByteArray(cudf::detail::hostdevice_span<PageInfo> pages,
                          cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                          size_t num_rows,
                          size_t min_row,
                          int level_type_size,
                          kernel_error::pointer error_code,
                          rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading the DELTA_LENGTH_BYTE_ARRAY column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeDeltaLengthByteArray(cudf::detail::hostdevice_span<PageInfo> pages,
                                cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                size_t num_rows,
                                size_t min_row,
                                int level_type_size,
                                kernel_error::pointer error_code,
                                rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading non-dictionary fixed width column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodePageDataFixed(cudf::detail::hostdevice_span<PageInfo> pages,
                         cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                         std::size_t num_rows,
                         size_t min_row,
                         int level_type_size,
                         kernel_error::pointer error_code,
                         rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading dictionary fixed width column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodePageDataFixedDict(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             std::size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             kernel_error::pointer error_code,
                             rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for reading dictionary fixed width column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in,out] pages All pages to be decoded
 * @param[in] chunks All chunks to be decoded
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read
 * @param[in] level_type_size Size in bytes of the type for level decoding
 * @param[out] error_code Error code for kernel failures
 * @param[in] stream CUDA stream to use
 */
void DecodeSplitPageDataFlat(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             std::size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             kernel_error::pointer error_code,
                             rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for initializing encoder row group fragments
 *
 * These fragments are used to calculate row group boundaries.
 * Based on the number of rows in each fragment, populates the value count, the size of data in the
 * fragment, the number of unique values, and the data size of unique values.
 *
 * @param[out] frag Fragment array [column_id][fragment_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] partitions Information about partitioning of table
 * @param[in] first_frag_in_part A Partition's offset into fragment array
 * @param[in] fragment_size Number of rows per fragment
 * @param[in] stream CUDA stream to use
 */
void InitRowGroupFragments(cudf::detail::device_2dspan<PageFragment> frag,
                           device_span<parquet_column_device_view const> col_desc,
                           device_span<partition_info const> partitions,
                           device_span<int const> first_frag_in_part,
                           uint32_t fragment_size,
                           rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for calculating encoder page fragments with variable fragment sizes
 *
 * Based on the number of rows in each fragment, populates the value count, the size of data in the
 * fragment, the number of unique values, and the data size of unique values.
 *
 * This assumes an initial call to InitRowGroupFragments has been made.
 *
 * @param[out] frag Fragment array [fragment_id]
 * @param[in] column_frag_sizes Number of rows per fragment per column [column_id]
 * @param[in] stream CUDA stream to use
 */
void CalculatePageFragments(device_span<PageFragment> frag,
                            device_span<size_type const> column_frag_sizes,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for initializing fragment statistics groups with variable fragment sizes
 *
 * @param[out] groups Statistics groups [total_fragments]
 * @param[in] fragments Page fragments [total_fragments]
 * @param[in] stream CUDA stream to use
 */
void InitFragmentStatistics(device_span<statistics_group> groups,
                            device_span<PageFragment const> fragments,
                            rmm::cuda_stream_view stream);

/**
 * @brief Initialize per-chunk hash maps used for dictionary with sentinel values
 *
 * @param chunks Flat span of chunks to initialize hash maps for
 * @param stream CUDA stream to use
 */
void initialize_chunk_hash_maps(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Insert chunk values into their respective hash maps
 *
 * @param frags Column fragments
 * @param stream CUDA stream to use
 */
void populate_chunk_hash_maps(cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream);

/**
 * @brief Compact dictionary hash map entries into chunk.dict_data
 *
 * @param chunks Flat span of chunks to compact hash maps for
 * @param stream CUDA stream to use
 */
void collect_map_entries(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Get the Dictionary Indices for each row
 *
 * For each row of a chunk, gets the indices into chunk.dict_data which contains the value otherwise
 * stored in input column [row]. Stores these indices into chunk.dict_index.
 *
 * Since dict_data itself contains indices into the original cudf column, this means that
 * col[row] == col[dict_data[dict_index[row - chunk.start_row]]]
 *
 * @param frags Column fragments
 * @param stream CUDA stream to use
 */
void get_dictionary_indices(cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for initializing encoder data pages
 *
 * @param[in,out] chunks Column chunks [rowgroup][column]
 * @param[out] pages Encode page array (null if just counting pages)
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_rowgroups Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] page_grstats Setup for page-level stats
 * @param[in] page_align Required alignment for uncompressed pages
 * @param[in] write_v2_headers True if V2 page headers should be written
 * @param[in] chunk_grstats Setup for chunk-level stats
 * @param[in] max_page_comp_data_size Calculated maximum compressed data size of pages
 * @param[in] stream CUDA stream to use
 */
void InitEncoderPages(cudf::detail::device_2dspan<EncColumnChunk> chunks,
                      device_span<EncPage> pages,
                      device_span<size_type> page_sizes,
                      device_span<size_type> comp_page_sizes,
                      device_span<parquet_column_device_view const> col_desc,
                      int32_t num_columns,
                      size_t max_page_size_bytes,
                      size_type max_page_size_rows,
                      uint32_t page_align,
                      bool write_v2_headers,
                      statistics_merge_group* page_grstats,
                      statistics_merge_group* chunk_grstats,
                      rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for packing column data into parquet pages
 *
 * If compression is to be used, `comp_in`, `comp_out`, and `comp_res` will be initialized for
 * use in subsequent compression operations.
 *
 * @param[in,out] pages Device array of EncPages (unordered)
 * @param[in] write_v2_headers True if V2 page headers should be written
 * @param[out] comp_in Compressor input buffers
 * @param[out] comp_out Compressor output buffers
 * @param[out] comp_res Compressor results
 * @param[in] stream CUDA stream to use
 */
void EncodePages(device_span<EncPage> pages,
                 bool write_v2_headers,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<compression_result> comp_res,
                 rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to make the compressed vs uncompressed chunk-level decision
 *
 * Also calculates the set of page encodings used for each chunk.
 *
 * @param[in,out] chunks Column chunks (updated with actual compressed/uncompressed sizes)
 * @param[in] stream CUDA stream to use
 */
void DecideCompression(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to encode page headers
 *
 * @param[in,out] pages Device array of EncPages
 * @param[in] comp_res Compressor status
 * @param[in] page_stats Optional page-level statistics to be included in page header
 * @param[in] chunk_stats Optional chunk-level statistics to be encoded
 * @param[in] stream CUDA stream to use
 */
void EncodePageHeaders(device_span<EncPage> pages,
                       device_span<compression_result const> comp_res,
                       device_span<statistics_chunk const> page_stats,
                       statistics_chunk const* chunk_stats,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to gather pages to a single contiguous block per chunk
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages
 * @param[in] stream CUDA stream to use
 */
void GatherPages(device_span<EncColumnChunk> chunks,
                 device_span<EncPage const> pages,
                 rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to calculate ColumnIndex information per chunk
 *
 * @param[in,out] chunks Column chunks
 * @param[in] column_stats Page-level statistics to be encoded
 * @param[in] column_index_truncate_length Max length of min/max values
 * @param[in] stream CUDA stream to use
 */
void EncodeColumnIndexes(device_span<EncColumnChunk> chunks,
                         device_span<statistics_chunk const> column_stats,
                         int32_t column_index_truncate_length,
                         rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::detail
