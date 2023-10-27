/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <text/bpe/byte_pair_encoding.cuh>

#include <nvtext/byte_pair_encoding.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/functional.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Loads a text file of merge-pairs into a strings column.
 *
 * The line position in the file indicates the pair's rank.
 *
 * @code{.pseudo}
 * Format of the file:
 * #version ..
 * a1 a2
 * b1 b2
 * c1 c2
 * ...
 * @endcode
 *
 * @param filename_merges Path to text file containing merge-pairs
 * @return object containing table elements for the BPE function
 */
std::unique_ptr<cudf::column> load_file_to_column(std::string const& filename_merges,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  std::ifstream merges_file(filename_merges);
  CUDF_EXPECTS(merges_file.good(), "Could not open " + filename_merges);

  std::vector<char> chars{};
  std::vector<cudf::size_type> offsets(1, 0);

  std::string line;
  std::getline(merges_file, line);
  std::string version = "#version";
  if (line.substr(0, version.size()).compare(version) == 0) { std::getline(merges_file, line); }

  // This is a text file delimited only by CR/LF.
  // TODO: Look into using the CSV reader to load the strings column instead.
  while (!line.empty()) {
    chars.insert(chars.end(), std::cbegin(line), std::cend(line));
    offsets.push_back(offsets.back() + line.length());
    std::getline(merges_file, line);
  }

  CUDF_EXPECTS(!chars.empty(), "No data found in " + filename_merges);

  auto d_chars   = cudf::detail::make_device_uvector_async(chars, stream, mr);
  auto d_offsets = cudf::detail::make_device_uvector_async(offsets, stream, mr);
  return cudf::make_strings_column(d_chars, d_offsets, {}, 0);
}

std::unique_ptr<detail::merge_pairs_map_type> initialize_merge_pairs_map(
  cudf::column_device_view const& input, rmm::cuda_stream_view stream)
{
  // Ensure capacity is at least (size/0.7) as documented here:
  // https://github.com/NVIDIA/cuCollections/blob/6ec8b6dcdeceea07ab4456d32461a05c18864411/include/cuco/static_map.cuh#L179-L182
  auto merge_pairs_map = std::make_unique<merge_pairs_map_type>(
    static_cast<size_t>(input.size() * 2),  // capacity is 2x;
    cuco::empty_key{-1},
    cuco::empty_value{-1},  // empty value is not used
    bpe_equal{input},
    probe_scheme{bpe_hasher{input}},
    hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); });

  merge_pairs_map->insert_async(iter, iter + input.size(), stream.value());

  return merge_pairs_map;
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& input, rmm::cuda_stream_view stream)
{
  auto d_input     = cudf::column_device_view::create(input->view(), stream);
  auto merge_pairs = initialize_merge_pairs_map(*d_input, stream);
  return std::make_unique<nvtext::bpe_merge_pairs::bpe_merge_pairs_impl>(
    std::move(input), std::move(d_input), std::move(merge_pairs));
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return create_bpe_merge_pairs_impl(std::make_unique<cudf::column>(input.parent(), stream, mr),
                                     stream);
}

}  // namespace

std::unique_ptr<bpe_merge_pairs> load_merge_pairs_file(std::string const& filename_merges,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  auto input_column = load_file_to_column(filename_merges, stream, mr);
  return std::make_unique<bpe_merge_pairs>(std::move(input_column), stream, mr);
}

}  // namespace detail

std::unique_ptr<bpe_merge_pairs> load_merge_pairs_file(std::string const& filename_merges,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::load_merge_pairs_file(filename_merges, cudf::get_default_stream(), mr);
}

bpe_merge_pairs::bpe_merge_pairs_impl::bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& merge_pairs,
  std::unique_ptr<cudf::column_device_view, std::function<void(cudf::column_device_view*)>>&&
    d_merge_pairs,
  std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map)
  : merge_pairs(std::move(merge_pairs)),
    d_merge_pairs(std::move(d_merge_pairs)),
    merge_pairs_map(std::move(merge_pairs_map))
{
}

bpe_merge_pairs::bpe_merge_pairs(std::unique_ptr<cudf::column>&& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource*)
  : impl(detail::create_bpe_merge_pairs_impl(std::move(input), stream))
{
}

bpe_merge_pairs::bpe_merge_pairs(cudf::strings_column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
  : impl(detail::create_bpe_merge_pairs_impl(input, stream, mr))
{
}

bpe_merge_pairs::~bpe_merge_pairs() = default;

}  // namespace nvtext
