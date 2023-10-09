/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <io/utilities/time_utils.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>

namespace cudf::io::parquet::detail {

namespace {

struct cumulative_row_info {
  size_t row_count;   // cumulative row count
  size_t size_bytes;  // cumulative size in bytes
  int key;            // schema index
};

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                rmm::device_uvector<int32_t> const& page_index,
                                rmm::device_uvector<cumulative_row_info> const& c_info,
                                rmm::cuda_stream_view stream)
{
  pages.device_to_host_sync(stream);

  printf("------------\nCumulative sizes by page\n");

  std::vector<int> schemas(pages.size());
  std::vector<int> h_page_index(pages.size());
  CUDF_CUDA_TRY(cudaMemcpy(
    h_page_index.data(), page_index.data(), sizeof(int) * pages.size(), cudaMemcpyDefault));
  std::vector<cumulative_row_info> h_cinfo(pages.size());
  CUDF_CUDA_TRY(cudaMemcpy(
    h_cinfo.data(), c_info.data(), sizeof(cumulative_row_info) * pages.size(), cudaMemcpyDefault));
  auto schema_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](size_type i) { return pages[h_page_index[i]].src_col_schema; });
  thrust::copy(thrust::seq, schema_iter, schema_iter + pages.size(), schemas.begin());
  auto last = thrust::unique(thrust::seq, schemas.begin(), schemas.end());
  schemas.resize(last - schemas.begin());
  printf("Num schemas: %lu\n", schemas.size());

  for (size_t idx = 0; idx < schemas.size(); idx++) {
    printf("Schema %d\n", schemas[idx]);
    for (size_t pidx = 0; pidx < pages.size(); pidx++) {
      auto const& page = pages[h_page_index[pidx]];
      if (page.flags & PAGEINFO_FLAGS_DICTIONARY || page.src_col_schema != schemas[idx]) {
        continue;
      }
      printf("\tP: {%lu, %lu}\n", h_cinfo[pidx].row_count, h_cinfo[pidx].size_bytes);
    }
  }
}

void print_cumulative_row_info(host_span<cumulative_row_info const> sizes,
                               std::string const& label,
                               std::optional<std::vector<chunk_read_info>> splits = std::nullopt)
{
  if (splits.has_value()) {
    printf("------------\nSplits\n");
    for (size_t idx = 0; idx < splits->size(); idx++) {
      printf("{%lu, %lu}\n", splits.value()[idx].skip_rows, splits.value()[idx].num_rows);
    }
  }

  printf("------------\nCumulative sizes %s\n", label.c_str());
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    printf("{%lu, %lu, %d}", sizes[idx].row_count, sizes[idx].size_bytes, sizes[idx].key);
    if (splits.has_value()) {
      // if we have a split at this row count and this is the last instance of this row count
      auto start = thrust::make_transform_iterator(
        splits->begin(), [](chunk_read_info const& i) { return i.skip_rows; });
      auto end               = start + splits->size();
      auto split             = std::find(start, end, sizes[idx].row_count);
      auto const split_index = [&]() -> int {
        if (split != end &&
            ((idx == sizes.size() - 1) || (sizes[idx + 1].row_count > sizes[idx].row_count))) {
          return static_cast<int>(std::distance(start, split));
        }
        return idx == 0 ? 0 : -1;
      }();
      if (split_index >= 0) {
        printf(" <-- split {%lu, %lu}",
               splits.value()[split_index].skip_rows,
               splits.value()[split_index].num_rows);
      }
    }
    printf("\n");
  }
}
#endif  // CHUNKING_DEBUG

/**
 * @brief Functor which reduces two cumulative_row_info structs of the same key.
 */
struct cumulative_row_sum {
  cumulative_row_info operator()
    __device__(cumulative_row_info const& a, cumulative_row_info const& b) const
  {
    return cumulative_row_info{a.row_count + b.row_count, a.size_bytes + b.size_bytes, a.key};
  }
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column.
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 */
struct row_size_functor {
  __device__ size_t validity_size(size_t num_rows, bool nullable)
  {
    return nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) * 4) : 0;
  }

  template <typename T>
  __device__ size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
{
  auto const offset_size = sizeof(size_type);
  // NOTE: Adding the + 1 offset here isn't strictly correct.  There will only be 1 extra offset
  // for the entire column, whereas this is adding an extra offset per page.  So we will get a
  // small over-estimate of the real size of the order :  # of pages * 4 bytes. It seems better
  // to overestimate size somewhat than to underestimate it and potentially generate chunks
  // that are too large.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

template <>
__device__ size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return validity_size(num_rows, nullable);
}

template <>
__device__ size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // only returns the size of offsets and validity. the size of the actual string chars
  // is tracked separately.
  auto const offset_size = sizeof(size_type);
  // see note about offsets in the list_view template.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all of
 * the data in this page.
 *
 * Sums across all nesting levels.
 */
struct get_cumulative_row_info {
  PageInfo const* const pages;

  __device__ cumulative_row_info operator()(size_type index)
  {
    auto const& page = pages[index];
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_row_info{0, 0, page.src_col_schema};
    }

    // total nested size, not counting string data
    auto iter =
      cudf::detail::make_counting_transform_iterator(0, [page, index] __device__(size_type i) {
        auto const& pni = page.nesting[i];
        return cudf::type_dispatcher(
          data_type{pni.type}, row_size_functor{}, pni.size, pni.nullable);
      });

    size_t const row_count = static_cast<size_t>(page.nesting[0].size);
    return {
      row_count,
      thrust::reduce(thrust::seq, iter, iter + page.num_output_nesting_levels) + page.str_bytes,
      page.src_col_schema};
  }
};

/**
 * @brief Functor which computes the effective size of all input columns by page.
 *
 * For a given row, we want to find the cost of all pages for all columns involved
 * in loading up to that row.  The complication here is that not all pages are the
 * same size between columns. Example:
 *
 *              page row counts
 * Column A:    0 <----> 100 <----> 200
 * Column B:    0 <---------------> 200 <--------> 400
                          |
 * if we decide to split at row 100, we don't really know the actual amount of bytes in column B
 * at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
 * page. Essentially, a conservative over-estimate of the real size.
 */
struct row_total_size {
  cumulative_row_info const* c_info;
  size_type const* key_offsets;
  size_t num_keys;

  __device__ cumulative_row_info operator()(cumulative_row_info const& i)
  {
    // sum sizes for each input column at this row
    size_t sum = 0;
    for (int idx = 0; idx < num_keys; idx++) {
      auto const start = key_offsets[idx];
      auto const end   = key_offsets[idx + 1];
      auto iter        = cudf::detail::make_counting_transform_iterator(
        0, [&] __device__(size_type i) { return c_info[i].row_count; });
      auto const page_index =
        thrust::lower_bound(thrust::seq, iter + start, iter + end, i.row_count) - iter;
      sum += c_info[page_index].size_bytes;
    }
    return {i.row_count, sum, i.key};
  }
};

/**
 * @brief Given a vector of cumulative {row_count, byte_size} pairs and a chunk read
 * limit, determine the set of splits.
 *
 * @param sizes Vector of cumulative {row_count, byte_size} pairs
 * @param num_rows Total number of rows to read
 * @param chunk_read_limit Limit on total number of bytes to be returned per read, for all columns
 */
std::vector<chunk_read_info> find_splits(std::vector<cumulative_row_info> const& sizes,
                                         size_t num_rows,
                                         size_t chunk_read_limit)
{
  // now we have an array of {row_count, real output bytes}. just walk through it and generate
  // splits.
  // TODO: come up with a clever way to do this entirely in parallel. For now, as long as batch
  // sizes are reasonably large, this shouldn't iterate too many times
  std::vector<chunk_read_info> splits;
  {
    size_t cur_pos             = 0;
    size_t cur_cumulative_size = 0;
    size_t cur_row_count       = 0;
    auto start = thrust::make_transform_iterator(sizes.begin(), [&](cumulative_row_info const& i) {
      return i.size_bytes - cur_cumulative_size;
    });
    auto end   = start + sizes.size();
    while (cur_row_count < num_rows) {
      int64_t split_pos =
        thrust::lower_bound(thrust::seq, start + cur_pos, end, chunk_read_limit) - start;

      // if we're past the end, or if the returned bucket is > than the chunk_read_limit, move back
      // one.
      if (static_cast<size_t>(split_pos) >= sizes.size() ||
          (sizes[split_pos].size_bytes - cur_cumulative_size > chunk_read_limit)) {
        split_pos--;
      }

      // best-try. if we can't find something that'll fit, we have to go bigger. we're doing this in
      // a loop because all of the cumulative sizes for all the pages are sorted into one big list.
      // so if we had two columns, both of which had an entry {1000, 10000}, that entry would be in
      // the list twice. so we have to iterate until we skip past all of them.  The idea is that we
      // either do this, or we have to call unique() on the input first.
      while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
             (split_pos < 0 || sizes[split_pos].row_count == cur_row_count)) {
        split_pos++;
      }

      auto const start_row = cur_row_count;
      cur_row_count        = sizes[split_pos].row_count;
      splits.push_back(chunk_read_info{start_row, cur_row_count - start_row});
      cur_pos             = split_pos;
      cur_cumulative_size = sizes[split_pos].size_bytes;
    }
  }
  // print_cumulative_row_info(sizes, "adjusted", splits);

  return splits;
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                                   type_id timestamp_type_id,
                                                                   Type physical,
                                                                   int8_t converted,
                                                                   int32_t length)
{
  int32_t type_width = (physical == FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_chrono(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  int8_t converted_type = converted;
  if (converted_type == DECIMAL && column_type_id != type_id::FLOAT64 &&
      not cudf::is_fixed_point(data_type{column_type_id})) {
    converted_type = UNKNOWN;  // Not converting to float64 or decimal
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

/**
 * @brief Return the required number of bits to store a value.
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

struct row_count_compare {
  __device__ bool operator()(cumulative_row_info const& a, cumulative_row_info const& b)
  {
    return a.row_count < b.row_count;
  }
};

}  // anonymous namespace

void reader::impl::create_global_chunk_info()
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(col.schema_idx);

      auto [type_width, clock_rate, converted_type] =
        conversion_info(to_type_id(schema, _strings_to_categorical, _timestamp_type.id()),
                        _timestamp_type.id(),
                        schema.type,
                        schema.converted_type,
                        schema.type_length);

      chunks.push_back(ColumnChunkDesc(col_meta.total_compressed_size,
                                       nullptr,
                                       col_meta.num_values,
                                       schema.type,
                                       type_width,
                                       row_group_start,
                                       row_group_rows,
                                       schema.max_definition_level,
                                       schema.max_repetition_level,
                                       _metadata->get_output_nesting_depth(col.schema_idx),
                                       required_bits(schema.max_definition_level),
                                       required_bits(schema.max_repetition_level),
                                       col_meta.codec,
                                       converted_type,
                                       schema.logical_type,
                                       schema.decimal_precision,
                                       clock_rate,
                                       i,
                                       col.schema_idx));
    }

    remaining_rows -= row_group_rows;
  }
}

void reader::impl::compute_input_passes()
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // if the user hasn't specified an input size limit, read everything in a single pass.
  if (_input_pass_read_limit == 0) {
    _file_itm_data.input_pass_row_group_offsets.push_back(0);
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    return;
  }

  // generate passes. make sure to account for the case where a single row group doesn't fit within
  //
  std::size_t const read_limit =
    _input_pass_read_limit > 0 ? _input_pass_read_limit : std::numeric_limits<std::size_t>::max();
  std::size_t cur_pass_byte_size = 0;
  std::size_t cur_rg_start       = 0;
  std::size_t cur_row_count      = 0;
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_row_count.push_back(0);

  for (size_t cur_rg_index = 0; cur_rg_index < row_groups_info.size(); cur_rg_index++) {
    auto const& rgi       = row_groups_info[cur_rg_index];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);

    // can we add this row group
    if (cur_pass_byte_size + row_group.total_byte_size >= read_limit) {
      // A single row group (the current one) is larger than the read limit:
      // We always need to include at least one row group, so end the pass at the end of the current
      // row group
      if (cur_rg_start == cur_rg_index) {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index + 1);
        _file_itm_data.input_pass_row_count.push_back(cur_row_count + row_group.num_rows);
        cur_rg_start       = cur_rg_index + 1;
        cur_pass_byte_size = 0;
      }
      // End the pass at the end of the previous row group
      else {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index);
        _file_itm_data.input_pass_row_count.push_back(cur_row_count);
        cur_rg_start       = cur_rg_index;
        cur_pass_byte_size = row_group.total_byte_size;
      }
    } else {
      cur_pass_byte_size += row_group.total_byte_size;
    }
    cur_row_count += row_group.num_rows;
  }
  // add the last pass if necessary
  if (_file_itm_data.input_pass_row_group_offsets.back() != row_groups_info.size()) {
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_row_count.push_back(cur_row_count);
  }
}

void reader::impl::setup_next_pass()
{
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<cudf::io::parquet::detail::pass_intermediate_data>();

  // setup row groups to be loaded for this pass
  auto const row_group_start = _file_itm_data.input_pass_row_group_offsets[_current_input_pass];
  auto const row_group_end   = _file_itm_data.input_pass_row_group_offsets[_current_input_pass + 1];
  auto const num_row_groups  = row_group_end - row_group_start;
  _pass_itm_data->row_groups.resize(num_row_groups);
  std::copy(_file_itm_data.row_groups.begin() + row_group_start,
            _file_itm_data.row_groups.begin() + row_group_end,
            _pass_itm_data->row_groups.begin());

  auto const num_passes = _file_itm_data.input_pass_row_group_offsets.size() - 1;
  CUDF_EXPECTS(_current_input_pass < num_passes, "Encountered an invalid read pass index");

  auto const chunks_per_rowgroup = _input_columns.size();
  auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

  auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
  auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

  _pass_itm_data->chunks = cudf::detail::hostdevice_vector<ColumnChunkDesc>(num_chunks, _stream);
  std::copy(chunk_start, chunk_end, _pass_itm_data->chunks.begin());

  // adjust skip_rows and num_rows by what's available in the row groups we are processing
  if (num_passes == 1) {
    _pass_itm_data->skip_rows = _file_itm_data.global_skip_rows;
    _pass_itm_data->num_rows  = _file_itm_data.global_num_rows;
  } else {
    auto const global_start_row = _file_itm_data.global_skip_rows;
    auto const global_end_row   = global_start_row + _file_itm_data.global_num_rows;
    auto const start_row =
      std::max(_file_itm_data.input_pass_row_count[_current_input_pass], global_start_row);
    auto const end_row =
      std::min(_file_itm_data.input_pass_row_count[_current_input_pass + 1], global_end_row);

    // skip_rows is always global in the sense that it is relative to the first row of
    // everything we will be reading, regardless of what pass we are on.
    // num_rows is how many rows we are reading this pass.
    _pass_itm_data->skip_rows =
      global_start_row + _file_itm_data.input_pass_row_count[_current_input_pass];
    _pass_itm_data->num_rows = end_row - start_row;
  }
}

void reader::impl::compute_splits_for_pass()
{
  auto const skip_rows = _pass_itm_data->skip_rows;
  auto const num_rows  = _pass_itm_data->num_rows;

  // simple case : no chunk size, no splits
  if (_output_chunk_read_limit <= 0) {
    _pass_itm_data->output_chunk_read_info = std::vector<chunk_read_info>{{skip_rows, num_rows}};
    return;
  }

  auto& pages = _pass_itm_data->pages_info;

  auto const& page_keys  = _pass_itm_data->page_keys;
  auto const& page_index = _pass_itm_data->page_index;

  // generate cumulative row counts and sizes
  rmm::device_uvector<cumulative_row_info> c_info(page_keys.size(), _stream);
  // convert PageInfo to cumulative_row_info
  auto page_input = thrust::make_transform_iterator(page_index.begin(),
                                                    get_cumulative_row_info{pages.device_ptr()});
  thrust::inclusive_scan_by_key(rmm::exec_policy(_stream),
                                page_keys.begin(),
                                page_keys.end(),
                                page_input,
                                c_info.begin(),
                                thrust::equal_to{},
                                cumulative_row_sum{});
  // print_cumulative_page_info(pages, page_index, c_info, stream);

  // sort by row count
  rmm::device_uvector<cumulative_row_info> c_info_sorted{c_info, _stream};
  thrust::sort(
    rmm::exec_policy(_stream), c_info_sorted.begin(), c_info_sorted.end(), row_count_compare{});

  // std::vector<cumulative_row_info> h_c_info_sorted(c_info_sorted.size());
  // CUDF_CUDA_TRY(cudaMemcpy(h_c_info_sorted.data(),
  //                          c_info_sorted.data(),
  //                          sizeof(cumulative_row_info) * c_info_sorted.size(),
  //                          cudaMemcpyDefault));
  // print_cumulative_row_info(h_c_info_sorted, "raw");

  // generate key offsets (offsets to the start of each partition of keys). worst case is 1 page per
  // key
  rmm::device_uvector<size_type> key_offsets(page_keys.size() + 1, _stream);
  auto const key_offsets_end = thrust::reduce_by_key(rmm::exec_policy(_stream),
                                                     page_keys.begin(),
                                                     page_keys.end(),
                                                     thrust::make_constant_iterator(1),
                                                     thrust::make_discard_iterator(),
                                                     key_offsets.begin())
                                 .second;
  size_t const num_unique_keys = key_offsets_end - key_offsets.begin();
  thrust::exclusive_scan(
    rmm::exec_policy(_stream), key_offsets.begin(), key_offsets.end(), key_offsets.begin());

  // adjust the cumulative info such that for each row count, the size includes any pages that span
  // that row count. this is so that if we have this case:
  //              page row counts
  // Column A:    0 <----> 100 <----> 200
  // Column B:    0 <---------------> 200 <--------> 400
  //                        |
  // if we decide to split at row 100, we don't really know the actual amount of bytes in column B
  // at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
  // page.
  //
  rmm::device_uvector<cumulative_row_info> aggregated_info(c_info.size(), _stream);
  thrust::transform(rmm::exec_policy(_stream),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    aggregated_info.begin(),
                    row_total_size{c_info.data(), key_offsets.data(), num_unique_keys});

  // bring back to the cpu
  std::vector<cumulative_row_info> h_aggregated_info(aggregated_info.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_aggregated_info.data(),
                                aggregated_info.data(),
                                sizeof(cumulative_row_info) * c_info.size(),
                                cudaMemcpyDefault,
                                _stream.value()));
  _stream.synchronize();

  // generate the actual splits
  _pass_itm_data->output_chunk_read_info =
    find_splits(h_aggregated_info, num_rows, _output_chunk_read_limit);
}

}  // namespace cudf::io::parquet::detail
