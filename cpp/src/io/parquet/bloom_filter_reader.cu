/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "arrow_filter_policy.cuh"
#include "compact_protocol_reader.hpp"
#include "io/parquet/parquet.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/logger.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <future>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>

namespace cudf::io::parquet::detail {

namespace {

std::pair<std::vector<char>, std::vector<size_t>> generate_chars_and_offsets(size_t num_keys)
{
  static std::vector<std::string> const strings{"first",
                                                "second",
                                                "third",
                                                "fourth",
                                                "fifth",
                                                "sixth",
                                                "seventh",
                                                "eighth",
                                                "ninth",
                                                "tenth",
                                                "eleventh",
                                                "twelfth",
                                                "thirteenth",
                                                "fourteenth",
                                                "fifteenth",
                                                "sixteenth",
                                                "seventeenth",
                                                "eighteenth"};

  auto constexpr seed = 0xf00d;
  /*static*/ std::mt19937 engine{seed};
  /*static*/ std::uniform_int_distribution dist{};

  std::vector<size_t> offsets(num_keys + 1);
  std::vector<char> chars;
  chars.reserve(12 * num_keys);  // 12 is the length of "seventeenth", the largest string
  size_t offset = 0;
  offsets[0]    = size_t{0};
  std::generate_n(offsets.begin() + 1, num_keys, [&]() {
    auto const& string  = strings[dist(engine) % strings.size()];
    auto const char_ptr = const_cast<char*>(string.data());
    chars.insert(chars.end(), char_ptr, char_ptr + string.length());
    offset += string.length();
    return offset;
  });
  return {std::move(chars), std::move(offsets)};
}

void validate_filter_bitwise(rmm::cuda_stream_view stream)
{
  std::size_t constexpr num_filter_blocks = 4;
  std::size_t constexpr num_keys          = 50;

  // Generate strings data
  auto const [h_chars, h_offsets] = generate_chars_and_offsets(num_keys);
  auto const mr                   = cudf::get_current_device_resource_ref();
  auto d_chars                    = cudf::detail::make_device_uvector_async(h_chars, stream, mr);
  auto d_offsets                  = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);

  rmm::device_uvector<cudf::string_view> d_keys(num_keys, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(num_keys),
                    d_keys.begin(),
                    [chars   = thrust::raw_pointer_cast(d_chars.data()),
                     offsets = thrust::raw_pointer_cast(d_offsets.data())] __device__(auto idx) {
                      return cudf::string_view{
                        chars + offsets[idx],
                        static_cast<cudf::size_type>(offsets[idx + 1] - offsets[idx])};
                    });

  using key_type    = cudf::string_view;
  using hasher_type = cudf::hashing::detail::XXHash_64<key_type>;
  using policy_type = cuco::arrow_filter_policy<key_type, hasher_type>;
  using filter_type = cuco::bloom_filter<key_type,
                                         cuco::extent<size_t>,
                                         cuda::thread_scope_device,
                                         policy_type,
                                         cudf::detail::cuco_allocator<char>>;
  // Spawn a bloom filter
  filter_type filter{
    num_filter_blocks,
    cuco::thread_scope_device,
    {hasher_type{0}},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream};

  // Add strings to the bloom filter
  filter.add(d_keys.begin(), d_keys.end(), stream);

  using word_type = filter_type::word_type;

  // Number of words in the filter
  size_t const num_words = filter.block_extent() * filter.words_per_block;

  // Get the filter bitset words
  cudf::device_span<word_type const> filter_bitset(filter.data(), num_words);
  // Expected filter bitset words
  rmm::device_vector<word_type> const expected_bitset{
    4194306,    4194305,  2359296,   1073774592, 524544,     1024,       268443648,  8519680,
    2147500040, 8421380,  269500416, 4202624,    8396802,    100665344,  2147747840, 5243136,
    131146,     655364,   285345792, 134222340,  545390596,  2281717768, 51201,      41943553,
    1619656708, 67441680, 8462730,   361220,     2216738864, 587333888,  4219272,    873463873};

  CUDF_EXPECTS(thrust::equal(rmm::exec_policy(stream),
                             expected_bitset.cbegin(),
                             expected_bitset.cend(),
                             filter_bitset.begin()),
               "Bloom filter bitset mismatched");
}

/**
 * @brief Temporary function that tests for key `third-037493` in bloom filters of column `c2` in
 * test Parquet file.
 *
 * @param buffer Device buffer containing bloom filter bitset
 * @param chunk Current row group index
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 */
void check_arbitrary_string_key(rmm::device_buffer const& buffer,
                                size_t chunk,
                                rmm::cuda_stream_view stream)
{
  using key_type    = cudf::string_view;
  using hasher_type = cudf::hashing::detail::XXHash_64<key_type>;
  using policy_type = cuco::arrow_filter_policy<key_type, hasher_type>;
  using word_type   = policy_type::word_type;

  auto constexpr word_size       = sizeof(word_type);
  auto constexpr words_per_block = policy_type::words_per_block;
  auto const num_filter_blocks   = buffer.size() / (word_size * words_per_block);

  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(1),
    [bitset     = const_cast<word_type*>(reinterpret_cast<word_type const*>(buffer.data())),
     num_blocks = num_filter_blocks,
     chunk      = chunk,
     stream     = stream] __device__(auto idx) {
      // using arrow_policy_type = cuco::arrow_filter_policy<key_type>;
      cuco::bloom_filter_ref<key_type,
                             cuco::extent<std::size_t>,
                             cuco::thread_scope_device,
                             policy_type>
        filter{bitset,
               num_blocks,
               {},  // scope
               {hasher_type{0}}};

      // literal to search
      cudf::string_view key("third-037493", 12);
      // Search in the filter
      if (filter.contains(key)) {
        printf("YES: Filter chunk: %lu contains key: third-037493\n", chunk);
      } else {
        printf("NO: Filter chunk: %lu does not contain key: third-037493\n", chunk);
      }
    });
}

/**
 * @brief Asynchronously reads bloom filters to device.
 *
 * @param sources Dataset sources
 * @param num_chunks Number of total column chunks to read
 * @param bloom_filter_data Devicebuffers to hold bloom filter bitsets for each chunk
 * @param bloom_filter_offsets Bloom filter offsets for all chunks
 * @param bloom_filter_sizes Bloom filter sizes for all chunks
 * @param chunk_source_map Association between each column chunk and its source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A future object for reading synchronization
 */
std::future<void> read_bloom_filters_async(
  host_span<std::unique_ptr<datasource> const> sources,
  size_t num_chunks,
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  cudf::host_span<std::optional<int64_t>> bloom_filter_offsets,
  cudf::host_span<std::optional<int32_t>> bloom_filter_sizes,
  std::vector<size_type> const& chunk_source_map,
  rmm::cuda_stream_view stream)
{
  // Read tasks for bloom filter data
  std::vector<std::future<size_t>> read_tasks;

  // Read bloom filters for all column chunks
  std::for_each(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_chunks),
    [&](auto const chunk) {
      // Read bloom filter if present
      if (bloom_filter_offsets[chunk].has_value()) {
        auto const bloom_filter_offset = bloom_filter_offsets[chunk].value();
        // If Bloom filter size (header + bitset) is available, just read the entire thing.
        // Else just read 256 bytes which will contain the entire header and may contain the
        // entire bitset as well.
        auto constexpr bloom_filter_header_size_guess = 256;
        auto const initial_read_size =
          bloom_filter_sizes[chunk].value_or(bloom_filter_header_size_guess);

        // Read an initial buffer from source
        auto& source = sources[chunk_source_map[chunk]];
        auto buffer  = source->host_read(bloom_filter_offset, initial_read_size);

        // Deserialize the Bloom filter header from the buffer.
        BloomFilterHeader header;
        CompactProtocolReader cp{buffer->data(), buffer->size()};
        cp.read(&header);

        // Test if header is valid.
        auto const is_header_valid =
          (header.num_bytes % 32) == 0 and
          header.compression.compression == BloomFilterCompression::Compression::UNCOMPRESSED and
          header.algorithm.algorithm == BloomFilterAlgorithm::Algorithm::SPLIT_BLOCK and
          header.hash.hash == BloomFilterHash::Hash::XXHASH;

        // Do not read if the bloom filter is invalid
        if (not is_header_valid) {
          CUDF_LOG_WARN("Encountered an invalid bloom filter header. Skipping");
          return;
        }

        // Bloom filter header size
        auto const bloom_filter_header_size = static_cast<int64_t>(cp.bytecount());
        auto const bitset_size              = header.num_bytes;

        // Check if we already read in the filter bitset in the initial read.
        if (initial_read_size >= bloom_filter_header_size + bitset_size) {
          bloom_filter_data[chunk] =
            rmm::device_buffer(buffer->data() + bloom_filter_header_size, bitset_size, stream);

          // MH: TODO: Temporary test. Remove me!!
          check_arbitrary_string_key(bloom_filter_data[chunk], chunk, stream);
        }
        // Read the bitset from datasource.
        else {
          auto const bitset_offset = bloom_filter_offset + bloom_filter_header_size;
          // Directly read to device if preferred
          if (source->is_device_read_preferred(bitset_size)) {
            bloom_filter_data[chunk] = rmm::device_buffer(bitset_size, stream);
            auto future_read_size =
              source->device_read_async(bitset_offset,
                                        bitset_size,
                                        static_cast<uint8_t*>(bloom_filter_data[chunk].data()),
                                        stream);

            read_tasks.emplace_back(std::move(future_read_size));
          } else {
            buffer                   = source->host_read(bitset_offset, bitset_size);
            bloom_filter_data[chunk] = rmm::device_buffer(buffer->data(), buffer->size(), stream);
          }
        }
      }
    });

  auto sync_fn = [](decltype(read_tasks) read_tasks) {
    for (auto& task : read_tasks) {
      task.wait();
    }
  };

  // MH: Remove me. Bitwise validation for bloom filter
  validate_filter_bitwise(stream);

  return std::async(std::launch::deferred, sync_fn, std::move(read_tasks));
}

/**
 * @brief Collects column indices with an equality predicate in the AST expression.
 * This is used in row group filtering based on bloom filters.
 */
class equality_predicate_converter : public ast::detail::expression_transformer {
 public:
  equality_predicate_converter(ast::expression const& expr, size_type const& num_columns)
    : _num_columns{num_columns}
  {
    expr.accept(*this);
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override
  {
    _equality_expr = std::reference_wrapper<ast::expression const>(expr);
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override
  {
    CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
                 "Equality AST supports only left table");
    CUDF_EXPECTS(expr.get_column_index() < _num_columns,
                 "Column index cannot be more than number of columns in the table");
    _equality_expr = std::reference_wrapper<ast::expression const>(expr);
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override
  {
    CUDF_FAIL("Column name reference is not supported in equality AST");
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    using cudf::ast::ast_operator;
    auto const operands = expr.get_operands();
    auto const op       = expr.get_operator();

    if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
      // First operand should be column reference, second should be literal.
      CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                   "Only binary operations are supported on column reference");
      CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      v->accept(*this);
      if (op == ast_operator::EQUAL) { equality_col_idx.emplace_back(v->get_column_index()); }
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2) {
        _operators.emplace_back(op, new_operands.front(), new_operands.back());
      } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
        _operators.emplace_back(op, new_operands.front());
      }
    }
    _equality_expr = std::reference_wrapper<ast::expression const>(_operators.back());
    return std::reference_wrapper<ast::expression const>(_operators.back());
  }

  /**
   * @brief Returns a list of column indices with an equality predicate in the AST expression.
   *
   * @return List of column indices
   */
  [[nodiscard]] std::vector<cudf::size_type> get_equality_col_idx() const
  {
    return equality_col_idx;
  }

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
  {
    std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
    for (auto const& operand : operands) {
      auto const new_operand = operand.get().accept(*this);
      transformed_operands.push_back(new_operand);
    }
    return transformed_operands;
  }
  std::optional<std::reference_wrapper<ast::expression const>> _equality_expr;
  std::vector<cudf::size_type> equality_col_idx;
  size_type _num_columns;
  std::list<ast::column_reference> _col_ref;
  std::list<ast::operation> _operators;
};

}  // namespace

std::optional<std::vector<std::vector<size_type>>> aggregate_reader_metadata::apply_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  auto const num_cols = output_dtypes.size();
  CUDF_EXPECTS(output_dtypes.size() == output_column_schemas.size(),
               "Mismatched size between lists of output column dtypes and output column schema");
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<column>> cols;
  // MH: How do I test for nested or non-comparable columns here?
  cols.emplace_back(cudf::make_numeric_column(
    data_type{cudf::type_id::INT32}, num_cols, rmm::device_buffer{}, 0, stream, mr));

  auto mutable_col_idx = cols.back()->mutable_view();

  thrust::sequence(rmm::exec_policy(stream),
                   mutable_col_idx.begin<size_type>(),
                   mutable_col_idx.end<size_type>(),
                   0);

  auto equality_table = cudf::table(std::move(cols));

  // Converts AST to EqualityAST with reference to min, max columns in above `stats_table`.
  equality_predicate_converter equality_expr{filter.get(), static_cast<size_type>(num_cols)};
  auto equality_col_schemas = equality_expr.get_equality_col_idx();

  // Convert column indices to column schema indices
  std::for_each(equality_col_schemas.begin(), equality_col_schemas.end(), [&](auto& col_idx) {
    col_idx = output_column_schemas[col_idx];
  });

  std::ignore = read_bloom_filters(sources, row_group_indices, equality_col_schemas, stream);

  return std::nullopt;
}

std::vector<rmm::device_buffer> aggregate_reader_metadata::read_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas,
  rmm::cuda_stream_view stream) const
{
  // Number of total row groups to process.
  auto const num_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_t{0},
    [](size_t sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = column_schemas.size();
  auto const num_chunks        = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<std::optional<int64_t>> bloom_filter_offsets(num_chunks);
  std::vector<std::optional<int32_t>> bloom_filter_sizes(num_chunks);

  // Gather all bloom filter offsets and sizes.
  size_type chunk_count = 0;

  // For all data sources
  std::for_each(thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator(row_group_indices.size()),
                [&](auto const src_index) {
                  // Get all row group indices in the data source
                  auto const& rg_indices = row_group_indices[src_index];
                  // For all row groups
                  std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
                    // For all column chunks
                    std::for_each(
                      column_schemas.begin(), column_schemas.end(), [&](auto const schema_idx) {
                        auto& col_meta = get_column_metadata(rg_index, src_index, schema_idx);

                        // Get bloom filter offsets and sizes
                        bloom_filter_offsets[chunk_count] = col_meta.bloom_filter_offset;
                        bloom_filter_sizes[chunk_count]   = col_meta.bloom_filter_length;

                        // Map each column chunk to its source index
                        chunk_source_map[chunk_count] = src_index;
                        chunk_count++;
                      });
                  });
                });

  // Do we have any bloom filters
  if (std::any_of(bloom_filter_offsets.cbegin(),
                  bloom_filter_offsets.cend(),
                  [](auto const offset) { return offset.has_value(); })) {
    // Create a vector to store bloom filter data
    std::vector<rmm::device_buffer> bloom_filter_data(num_chunks);

    // Wait on bloom filter read tasks
    read_bloom_filters_async(sources,
                             num_chunks,
                             bloom_filter_data,
                             bloom_filter_offsets,
                             bloom_filter_sizes,
                             chunk_source_map,
                             stream)
      .wait();
    // Return the vector
    return bloom_filter_data;
  }
  return {};
}

}  // namespace cudf::io::parquet::detail
