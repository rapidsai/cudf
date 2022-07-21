/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "io/comp/gpuinflate.hpp"
#include "io/parquet/parquet.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/statistics/statistics.cuh"
#include "io/utilities/column_buffer.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuco/static_map.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <vector>

namespace cudf {
namespace io {
namespace parquet {

using cudf::io::detail::string_index_pair;

// Total number of unsigned 16 bit values
constexpr size_type MAX_DICT_SIZE =
  std::numeric_limits<uint16_t>::max() - std::numeric_limits<uint16_t>::min() + 1;

/**
 * @brief Struct representing an input column in the file.
 */
struct input_column_info {
  int schema_idx;
  std::string name;
  // size == nesting depth. the associated real output
  // buffer index in the dest column for each level of nesting.
  std::vector<int> nesting;
  auto nesting_depth() const { return nesting.size(); }
};

namespace gpu {

auto constexpr KEY_SENTINEL   = size_type{-1};
auto constexpr VALUE_SENTINEL = size_type{-1};
using map_type                = cuco::static_map<size_type, size_type>;
using slot_type               = map_type::pair_atomic_type;

/**
 * @brief Enums for the flags in the page header
 */
enum {
  PAGEINFO_FLAGS_DICTIONARY = (1 << 0),  // Indicates a dictionary page
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
 * @brief Nesting information
 */
struct PageNestingInfo {
  // input repetition/definition levels are remapped with these values
  // into the corresponding real output nesting depths.
  int32_t start_depth;
  int32_t end_depth;

  // set at initialization
  int32_t max_def_level;
  int32_t max_rep_level;

  // set during preprocessing
  int32_t size;              // this page/nesting-level's size contribution to the output column
  int32_t page_start_value;  // absolute output start index in output column data

  // set during data decoding
  int32_t valid_count;       // # of valid values decoded in this page/nesting-level
  int32_t value_count;       // total # of values decoded in this page/nesting-level
  int32_t null_count;        // null count
  int32_t valid_map_offset;  // current offset in bits relative to valid_map
  uint8_t* data_out;         // pointer into output buffer
  uint32_t* valid_map;       // pointer into output validity buffer
};

/**
 * @brief Struct describing a particular page of column chunk data
 */
struct PageInfo {
  uint8_t* page_data;  // Compressed page data before decompression, or uncompressed data after
                       // decompression
  int32_t compressed_page_size;    // compressed data size in bytes
  int32_t uncompressed_page_size;  // uncompressed data size in bytes
  // Number of values in this data page or dictionary.
  // Important : the # of input values does not necessarily
  // correspond to the number of rows in the output. It just reflects the number
  // of values in the input stream.
  // - In the case of a flat schema, it will correspond to the # of output rows
  // - In the case of a nested schema, you have to decode the repetition and definition
  //   levels to extract actual column values
  int32_t num_input_values;
  int32_t chunk_row;       // starting row of this page relative to the start of the chunk
  int32_t num_rows;        // number of rows in this page
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
  int skipped_values;
  // # of values skipped in the actual data stream.
  int skipped_leaf_values;

  // nesting information (input/output) for each page
  int num_nesting_levels;
  PageNestingInfo* nesting;
};

/**
 * @brief Struct describing a particular chunk of column data
 */
struct ColumnChunkDesc {
  ColumnChunkDesc() = default;
  explicit ColumnChunkDesc(size_t compressed_size_,
                           uint8_t* compressed_data_,
                           size_t num_values_,
                           uint16_t datatype_,
                           uint16_t datatype_length_,
                           size_t start_row_,
                           uint32_t num_rows_,
                           int16_t max_definition_level_,
                           int16_t max_repetition_level_,
                           int16_t max_nesting_depth_,
                           uint8_t def_level_bits_,
                           uint8_t rep_level_bits_,
                           int8_t codec_,
                           int8_t converted_type_,
                           LogicalType logical_type_,
                           int8_t decimal_scale_,
                           int32_t ts_clock_rate_,
                           int32_t src_col_index_,
                           int32_t src_col_schema_)
    : compressed_data(compressed_data_),
      compressed_size(compressed_size_),
      num_values(num_values_),
      start_row(start_row_),
      num_rows(num_rows_),
      max_level{max_definition_level_, max_repetition_level_},
      max_nesting_depth{max_nesting_depth_},
      data_type(datatype_ | (datatype_length_ << 3)),
      level_bits{def_level_bits_, rep_level_bits_},
      num_data_pages(0),
      num_dict_pages(0),
      max_num_pages(0),
      page_info(nullptr),
      str_dict_index(nullptr),
      valid_map_base{nullptr},
      column_data_base{nullptr},
      codec(codec_),
      converted_type(converted_type_),
      logical_type(logical_type_),
      decimal_scale(decimal_scale_),
      ts_clock_rate(ts_clock_rate_),
      src_col_index(src_col_index_),
      src_col_schema(src_col_schema_)
  {
  }

  uint8_t const* compressed_data;                  // pointer to compressed column chunk data
  size_t compressed_size;                          // total compressed data size for this chunk
  size_t num_values;                               // total number of values in this column
  size_t start_row;                                // starting row of this chunk
  uint32_t num_rows;                               // number of rows in this chunk
  int16_t max_level[level_type::NUM_LEVEL_TYPES];  // max definition/repetition level
  int16_t max_nesting_depth;                       // max nesting depth of the output
  uint16_t data_type;                              // basic column data type, ((type_length << 3) |
                                                   // parquet::Type)
  uint8_t
    level_bits[level_type::NUM_LEVEL_TYPES];  // bits to encode max definition/repetition levels
  int32_t num_data_pages;                     // number of data pages
  int32_t num_dict_pages;                     // number of dictionary pages
  int32_t max_num_pages;                      // size of page_info array
  PageInfo* page_info;                        // output page info for up to num_dict_pages +
                                              // num_data_pages (dictionary pages first)
  string_index_pair* str_dict_index;          // index for string dictionary
  uint32_t** valid_map_base;                  // base pointers of valid bit map for this column
  void** column_data_base;                    // base pointers of column data
  int8_t codec;                               // compressed codec enum
  int8_t converted_type;                      // converted type enum
  LogicalType logical_type;                   // logical type
  int8_t decimal_scale;                       // decimal scale pow(10, -decimal_scale)
  int32_t ts_clock_rate;  // output timestamp clock frequency (0=default, 1000=ms, 1000000000=ns)

  int32_t src_col_index;   // my input column index
  int32_t src_col_schema;  // my schema index in the file
};

/**
 * @brief Struct describing an encoder column
 */
struct parquet_column_device_view : stats_column_desc {
  Type physical_type;      //!< physical data type
  uint8_t converted_type;  //!< logical data type
  uint8_t level_bits;  //!< bits to encode max definition (lower nibble) & repetition (upper nibble)
                       //!< levels
  constexpr uint8_t num_def_level_bits() { return level_bits & 0xf; }
  constexpr uint8_t num_rep_level_bits() { return level_bits >> 4; }
  size_type const* const*
    nesting_offsets;  //!< If column is a nested type, contains offset array of each nesting level

  size_type const* level_offsets;  //!< Offset array for per-row pre-calculated rep/def level values
  uint8_t const* rep_values;       //!< Pre-calculated repetition level values
  uint8_t const* def_values;       //!< Pre-calculated definition level values
  uint8_t const* nullability;  //!< Array of nullability of each nesting level. e.g. nullable[0] is
                               //!< nullability of parent_column. May be different from
                               //!< col.nullable() in case of chunked writing.
};

constexpr int max_page_fragment_size = 5000;  //!< Max number of rows in a page fragment

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
  size_type start_row;       //!< First row in fragment
  uint16_t num_rows;         //!< Number of rows in fragment
  uint16_t num_dict_vals;    //!< Number of unique dictionary entries
  EncColumnChunk* chunk;     //!< The chunk that this fragment belongs to
};

/// Size of hash used for building dictionaries
constexpr unsigned int kDictHashBits = 16;
constexpr size_t kDictScratchSize    = (1 << kDictHashBits) * sizeof(uint32_t);

/**
 * @brief Return the byte length of parquet dtypes that are physically represented by INT32
 */
inline uint32_t __device__ int32_logical_len(type_id id)
{
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8: return 1;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: return 2;
    default: return 4;
  }
}

/**
 * @brief Translate the row index of a parent column_device_view into the index of the first value
 * in the leaf child.
 * Only works in the context of parquet writer where struct columns are previously modified s.t.
 * they only have one immediate child.
 */
inline size_type __device__ row_to_value_idx(size_type idx, column_device_view col)
{
  while (col.type().id() == type_id::LIST or col.type().id() == type_id::STRUCT) {
    if (col.type().id() == type_id::STRUCT) {
      idx += col.offset();
      col = col.child(0);
    } else {
      auto list_col = cudf::detail::lists_column_device_view(col);
      idx           = list_col.offset_at(idx);
      col           = list_col.child();
    }
  }
  return idx;
}

/**
 * @brief Return worst-case compressed size of compressed data given the uncompressed size
 */
inline size_t __device__ __host__ GetMaxCompressedBfrSize(size_t uncomp_size,
                                                          uint32_t num_pages = 1)
{
  return uncomp_size + (uncomp_size >> 7) + num_pages * 8;
}

struct EncPage;

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
  uint16_t* dict_index;   //!< Index of value in dictionary page. column[dict_data[dict_index[row]]]
  uint8_t dict_rle_bits;  //!< Bit size for encoding dictionary indices
  bool use_dictionary;    //!< True if the chunk uses dictionary encoding
};

/**
 * @brief Struct describing an encoder data page
 */
struct EncPage {
  uint8_t* page_data;        //!< Ptr to uncompressed page
  uint8_t* compressed_data;  //!< Ptr to compressed page
  uint16_t num_fragments;    //!< Number of fragments in page
  PageType page_type;        //!< Page type
  EncColumnChunk* chunk;     //!< Chunk that this page belongs to
  uint32_t chunk_id;         //!< Index in chunk array
  uint32_t hdr_size;         //!< Size of page header
  uint32_t max_hdr_size;     //!< Maximum size of page header
  uint32_t max_data_size;    //!< Maximum size of coded page data (excluding header)
  uint32_t start_row;        //!< First row of page
  uint32_t num_rows;         //!< Rows in page
  uint32_t num_leaf_values;  //!< Values in page. Different from num_rows in case of nested types
  uint32_t num_values;  //!< Number of def/rep level values in page. Includes null/empty elements in
                        //!< non-leaf levels
  decompress_status* comp_stat;  //!< Ptr to compression status
};

/**
 * @brief Launches kernel for parsing the page headers in the column chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 */
void DecodePageHeaders(ColumnChunkDesc* chunks, int32_t num_chunks, rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for building the dictionary index for the column
 * chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 */
void BuildStringDictionaryIndex(ColumnChunkDesc* chunks,
                                int32_t num_chunks,
                                rmm::cuda_stream_view stream);

/**
 * @brief Preprocess column information for nested schemas.
 *
 * There are several pieces of information we can't compute directly from row counts in
 * the parquet headers when dealing with nested schemas.
 * - The total sizes of all output columns at all nesting levels
 * - The starting output buffer offset for each page, for each nesting level
 * For flat schemas, these values are computed during header decoding (see gpuDecodePageHeaders)
 *
 * Note : this function is where output device memory is allocated for nested columns.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param input_columns Input column information
 * @param output_columns Output column information
 * @param num_rows Maximum number of rows to read
 * @param min_rows crop all rows below min_row
 * @param uses_custom_row_bounds Whether or not num_rows and min_rows represents user-specific
 * bounds
 * @param stream Cuda stream
 */
void PreprocessColumnData(hostdevice_vector<PageInfo>& pages,
                          hostdevice_vector<ColumnChunkDesc> const& chunks,
                          std::vector<input_column_info>& input_columns,
                          std::vector<cudf::io::detail::column_buffer>& output_columns,
                          size_t num_rows,
                          size_t min_row,
                          bool uses_custom_row_bounds,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr);

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
 * @param[in] stream CUDA stream to use, default 0
 */
void DecodePageData(hostdevice_vector<PageInfo>& pages,
                    hostdevice_vector<ColumnChunkDesc> const& chunks,
                    size_t num_rows,
                    size_t min_row,
                    rmm::cuda_stream_view stream);

/**
 * @brief Dremel data that describes one nested type column
 *
 * @see get_dremel_data()
 */
struct dremel_data {
  rmm::device_uvector<size_type> dremel_offsets;
  rmm::device_uvector<uint8_t> rep_level;
  rmm::device_uvector<uint8_t> def_level;

  size_type leaf_data_size;
};

/**
 * @brief Get the dremel offsets and repetition and definition levels for a LIST column
 *
 * Dremel offsets are the per row offsets into the repetition and definition level arrays for a
 * column.
 * Example:
 * ```
 * col            = {{1, 2, 3}, { }, {5, 6}}
 * dremel_offsets = { 0,         3,   4,  6}
 * rep_level      = { 0, 1, 1,   0,   0, 1}
 * def_level      = { 1, 1, 1,   0,   1, 1}
 * ```
 * @param col Column of LIST type
 * @param level_nullability Pre-determined nullability at each list level. Empty means infer from
 * `col`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A struct containing dremel data
 */
dremel_data get_dremel_data(column_view h_col,
                            rmm::device_uvector<uint8_t> const& d_nullability,
                            std::vector<uint8_t> const& nullability,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for initializing encoder page fragments
 *
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
void InitPageFragments(cudf::detail::device_2dspan<PageFragment> frag,
                       device_span<parquet_column_device_view const> col_desc,
                       device_span<partition_info const> partitions,
                       device_span<int const> first_frag_in_part,
                       uint32_t fragment_size,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for initializing fragment statistics groups
 *
 * @param[out] groups Statistics groups [num_columns x num_fragments]
 * @param[in] fragments Page fragments [num_columns x num_fragments]
 * @param[in] col_desc Column description [num_columns]
 * @param[in] stream CUDA stream to use
 */
void InitFragmentStatistics(cudf::detail::device_2dspan<statistics_group> groups,
                            cudf::detail::device_2dspan<PageFragment const> fragments,
                            device_span<gpu::parquet_column_device_view const> col_desc,
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
void populate_chunk_hash_maps(cudf::detail::device_2dspan<gpu::PageFragment const> frags,
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
void get_dictionary_indices(cudf::detail::device_2dspan<gpu::PageFragment const> frags,
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
 * @param[in] chunk_grstats Setup for chunk-level stats
 * @param[in] max_page_comp_data_size Calculated maximum compressed data size of pages
 * @param[in] stream CUDA stream to use, default 0
 */
void InitEncoderPages(cudf::detail::device_2dspan<EncColumnChunk> chunks,
                      device_span<gpu::EncPage> pages,
                      device_span<size_type> page_sizes,
                      device_span<size_type> comp_page_sizes,
                      device_span<parquet_column_device_view const> col_desc,
                      int32_t num_columns,
                      size_t max_page_size_bytes,
                      size_type max_page_size_rows,
                      statistics_merge_group* page_grstats,
                      statistics_merge_group* chunk_grstats,
                      rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for packing column data into parquet pages
 *
 * @param[in,out] pages Device array of EncPages (unordered)
 * @param[out] comp_in Compressor input buffers
 * @param[out] comp_in Compressor output buffers
 * @param[out] comp_stats Compressor statuses
 * @param[in] stream CUDA stream to use, default 0
 */
void EncodePages(device_span<EncPage> pages,
                 device_span<device_span<uint8_t const>> comp_in,
                 device_span<device_span<uint8_t>> comp_out,
                 device_span<decompress_status> comp_stats,
                 rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to make the compressed vs uncompressed chunk-level decision
 *
 * @param[in,out] chunks Column chunks (updated with actual compressed/uncompressed sizes)
 * @param[in] stream CUDA stream to use, default 0
 */
void DecideCompression(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to encode page headers
 *
 * @param[in,out] pages Device array of EncPages
 * @param[in] comp_stats Compressor status
 * @param[in] page_stats Optional page-level statistics to be included in page header
 * @param[in] chunk_stats Optional chunk-level statistics to be encoded
 * @param[in] stream CUDA stream to use, default 0
 */
void EncodePageHeaders(device_span<EncPage> pages,
                       device_span<decompress_status const> comp_stats,
                       device_span<statistics_chunk const> page_stats,
                       const statistics_chunk* chunk_stats,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel to gather pages to a single contiguous block per chunk
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages
 * @param[in] stream CUDA stream to use, default 0
 */
void GatherPages(device_span<EncColumnChunk> chunks,
                 device_span<gpu::EncPage const> pages,
                 rmm::cuda_stream_view stream);

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
