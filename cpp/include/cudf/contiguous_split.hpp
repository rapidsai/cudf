/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {

/**
 * @addtogroup copy_split
 * @{
 * @file
 * @brief Table APIs for contiguous_split, pack, unpack, and metadata
 */

/**
 * @brief Column data in a serialized format
 *
 * Contains data from an array of columns in two contiguous buffers: one on host, which contains
 * table metadata and one on device which contains the table data.
 */
struct packed_columns {
  packed_columns()
    : metadata(std::make_unique<std::vector<uint8_t>>()),
      gpu_data(std::make_unique<rmm::device_buffer>())
  {
  }

  /**
   * @brief Construct a new packed columns object
   *
   * @param md Host-side metadata buffer
   * @param gd Device-side data buffer
   */
  packed_columns(std::unique_ptr<std::vector<uint8_t>>&& md,
                 std::unique_ptr<rmm::device_buffer>&& gd)
    : metadata(std::move(md)), gpu_data(std::move(gd))
  {
  }

  std::unique_ptr<std::vector<uint8_t>> metadata;  ///< Host-side metadata buffer
  std::unique_ptr<rmm::device_buffer> gpu_data;    ///< Device-side data buffer
};

/**
 * @brief The result(s) of a cudf::contiguous_split
 *
 * Each table_view resulting from a split operation performed by contiguous_split,
 * will be returned wrapped in a `packed_table`. The table_view and internal
 * column_views in this struct are not owned by a top level cudf::table or cudf::column.
 * The backing memory and metadata is instead owned by the `data` field and is in one
 * contiguous block.
 *
 * The user is responsible for assuring that the `table` or any derived table_views do
 * not outlive the memory owned by `data`.
 */
struct packed_table {
  cudf::table_view table;  ///< Result table_view of a cudf::contiguous_split
  packed_columns data;     ///< Column data owned
};

/**
 * @brief Performs a deep-copy split of a `table_view` into a vector of `packed_table` where each
 * `packed_table` is using a single contiguous block of memory for all of the split's column data.
 *
 * The memory for the output views is allocated in a single contiguous `rmm::device_buffer` returned
 * in the `packed_table`. There is no top-level owning table.
 *
 * The returned views of `input` are constructed from a vector of indices, that indicate
 * where each split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`.
 * For a `splits` size N, there will always be N+1 splits in the output.
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory contained in the `all_data` field of the
 * returned packed_table.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 *
 * @throws std::out_of_range if `splits` has end index > size of `input`.
 * @throws std::out_of_range When the value in `splits` is not in the range [0, input.size()).
 * @throws std::invalid_argument When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits A vector of indices where the view will be split
 * @param mr An optional memory resource to use for all returned device allocations
 * @return The set of requested views of `input` indicated by the `splits` and the viewed memory
 * buffer
 */
std::vector<packed_table> contiguous_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

namespace detail {
struct contiguous_split_state;
};

/**
 * @brief Perform a chunked "pack" operation of the input `table_view` using a user provided
 * buffer of size `user_buffer_size`.
 *
 * The intent of this operation is to be used in a streamed fashion at times of GPU
 * out-of-memory, where we want to minimize the number of small cudaMemcpy calls and
 * tracking of all the metadata associated with cudf tables. Because of the memory constraints,
 * all thrust and scratch memory allocations are using the passed-in memory resource exclusively,
 * not a per-device memory resource.
 *
 * This class defines two methods that must be used in concert to carry out the chunked_pack:
 * has_next and next. Here is an example:
 *
 * @code{.pseudo}
 * // Create a table_view
 * cudf::table_view tv = ...;
 *
 * // Choose a memory resource (optional). This memory resource is used for scratch/thrust temporary
 * // data. In memory constrained cases, this can be used to set aside scratch memory
 * // for `chunked_pack` at the beginning of a program.
 * auto mr = rmm::mr::get_current_device_resource();
 *
 * // Define a buffer size for each chunk: the larger the buffer is, the more SMs can be
 * // occupied by this algorithm.
 * //
 * // Internally, the GPU unit of work is a 1MB batch. When we instantiate `cudf::chunked_pack`,
 * // all the 1MB batches for the source table_view are computed up front. Additionally,
 * // chunked_pack calculates the number of iterations that are required to go through all those
 * // batches given a `user_buffer_size` buffer. The number of 1MB batches in each iteration (chunk)
 * // equals the number of CUDA blocks that will be used for the main kernel launch.
 * //
 * std::size_t user_buffer_size = 128*1024*1024;
 *
 * auto chunked_packer = cudf::chunked_pack::create(tv, user_buffer_size, mr);
 *
 * std::size_t host_offset = 0;
 * auto host_buffer = ...; // obtain a host buffer you would like to copy to
 *
 * while (chunked_packer->has_next()) {
 *   // get a user buffer of size `user_buffer_size`
 *   cudf::device_span<uint8_t> user_buffer = ...;
 *   std::size_t bytes_copied = chunked_packer->next(user_buffer);
 *
 *   // buffer will hold the contents of at most `user_buffer_size` bytes
 *   // of the contiguously packed input `table_view`. You are now free to copy
 *   // this memory somewhere else, for example, to host.
 *   cudaMemcpyAsync(
 *     host_buffer.data() + host_offset,
 *     user_buffer.data(),
 *     bytes_copied,
 *     cudaMemcpyDefault,
 *     stream);
 *
 *   host_offset += bytes_copied;
 * }
 * @endcode
 */
class chunked_pack {
 public:
  /**
   * @brief Construct a `chunked_pack` class.
   *
   * @param input source `table_view` to pack
   * @param user_buffer_size buffer size (in bytes) that will be passed on `next`. Must be
   *                         at least 1MB
   * @param temp_mr An optional memory resource to be used for temporary and scratch allocations
   * only
   */
  explicit chunked_pack(
    cudf::table_view const& input,
    std::size_t user_buffer_size,
    rmm::mr::device_memory_resource* temp_mr = rmm::mr::get_current_device_resource());

  /**
   * @brief Destructor that will be implemented as default. Declared with definition here because
   * contiguous_split_state is incomplete at this stage.
   */
  ~chunked_pack();

  /**
   * @brief Obtain the total size of the contiguously packed `table_view`.
   *
   * @return total size (in bytes) of all the chunks
   */
  [[nodiscard]] std::size_t get_total_contiguous_size() const;

  /**
   * @brief Function to check if there are chunks left to be copied.
   *
   * @return true if there are chunks left to be copied, and false otherwise
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Packs the next chunk into `user_buffer`. This should be called as long as
   * `has_next` returns true. If `next` is called when `has_next` is false, an exception
   * is thrown.
   *
   * @throws cudf::logic_error If the size of `user_buffer` is different than `user_buffer_size`
   * @throws cudf::logic_error If called after all chunks have been copied
   *
   * @param user_buffer device span target for the chunk. The size of this span must equal
   *                    the `user_buffer_size` parameter passed at construction
   * @return The number of bytes that were written to `user_buffer` (at most
   *          `user_buffer_size`)
   */
  [[nodiscard]] std::size_t next(cudf::device_span<uint8_t> const& user_buffer);

  /**
   * @brief Build the opaque metadata for all added columns.
   *
   * @return A vector containing the serialized column metadata
   */
  [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> build_metadata() const;

  /**
   * @brief Creates a `chunked_pack` instance to perform a "pack" of the `table_view`
   * "input", where a buffer of `user_buffer_size` is filled with chunks of the
   * overall operation. This operation can be used in cases where GPU memory is constrained.
   *
   * The memory resource (`temp_mr`) could be a special memory resource to be used in
   * situations when GPU memory is low and we want scratch and temporary allocations to
   * happen from a small reserved pool of memory. Note that it defaults to the regular cuDF
   * per-device resource.
   *
   * @throws cudf::logic_error When user_buffer_size is less than 1MB
   *
   * @param input source `table_view` to pack
   * @param user_buffer_size buffer size (in bytes) that will be passed on `next`. Must be
   *                         at least 1MB
   * @param temp_mr RMM memory resource to be used for temporary and scratch allocations only
   * @return a unique_ptr of chunked_pack
   */
  [[nodiscard]] static std::unique_ptr<chunked_pack> create(
    cudf::table_view const& input,
    std::size_t user_buffer_size,
    rmm::mr::device_memory_resource* temp_mr = rmm::mr::get_current_device_resource());

 private:
  // internal state of contiguous split
  std::unique_ptr<detail::contiguous_split_state> state;
};

/**
 * @brief Deep-copy a `table_view` into a serialized contiguous memory format.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes and the data from the
 * `table_view` is copied into a `device_buffer`. Pass the output of this function into
 * `cudf::unpack` to deserialize.
 *
 * @param input View of the table to pack
 * @param mr An optional memory resource to use for all returned device allocations
 * @return packed_columns A struct containing the serialized metadata and data in contiguous host
 *         and device memory respectively
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Produce the metadata used for packing a table stored in a contiguous buffer.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes which can be used to
 * construct a `packed_columns` or `packed_table` structure. The caller is responsible for
 * guaranteeing that all of the columns in the table point into `contiguous_buffer`.
 *
 * @param table View of the table to pack
 * @param contiguous_buffer A contiguous buffer of device memory which contains the data referenced
 *        by the columns in `table`
 * @param buffer_size The size of `contiguous_buffer`
 * @return Vector of bytes representing the metadata used to `unpack` a packed_columns struct
 */
std::vector<uint8_t> pack_metadata(table_view const& table,
                                   uint8_t const* contiguous_buffer,
                                   size_t buffer_size);

/**
 * @brief Deserialize the result of `cudf::pack`.
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `input`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param input The packed columns to unpack
 * @return The unpacked `table_view`
 */
table_view unpack(packed_columns const& input);

/**
 * @brief Deserialize the result of `cudf::pack`.
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `gpu_data` using the metadata contained in the host
 * buffer `metadata`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param metadata The host-side metadata buffer resulting from the initial pack() call
 * @param gpu_data The device-side contiguous buffer storing the data that will be referenced by
 *        the resulting `table_view`
 * @return The unpacked `table_view`
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data);

/** @} */
}  // namespace cudf
