/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "compact_protocol_reader.hpp"
#include "io/parquet/parquet.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/logger.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>

#include <cuco/bloom_filter_policies.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include <future>
#include <numeric>
#include <optional>

namespace cudf::io::parquet::detail {
namespace {

/**
 * @brief Converts bloom filter membership results (for each column chunk) to a device column.
 *
 */
struct bloom_filter_caster {
  cudf::device_span<cudf::device_span<cuda::std::byte> const> bloom_filter_spans;
  host_span<Type const> parquet_types;
  size_t total_row_groups;
  size_t num_equality_columns;

  enum class is_int96_timestamp : bool { YES, NO };

  template <typename T, is_int96_timestamp IS_INT96_TIMESTAMP = is_int96_timestamp::NO>
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                   std::unique_ptr<cudf::column>>
  query_bloom_filter(cudf::size_type equality_col_idx,
                     cudf::data_type dtype,
                     ast::literal const* const literal,
                     rmm::cuda_stream_view stream) const
  {
    using key_type          = T;
    using policy_type       = cuco::arrow_filter_policy<key_type, cudf::hashing::detail::XXHash_64>;
    using bloom_filter_type = cuco::
      bloom_filter_ref<key_type, cuco::extent<std::size_t>, cuco::thread_scope_thread, policy_type>;
    using filter_block_type = typename bloom_filter_type::filter_block_type;
    using word_type         = typename policy_type::word_type;

    // Check if the literal has the same type as the predicate column
    CUDF_EXPECTS(
      dtype == literal->get_data_type() and
        cudf::have_same_types(
          cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
          cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
      "Mismatched predicate column and literal types");

    // Filter properties
    auto constexpr bytes_per_block = sizeof(word_type) * policy_type::words_per_block;

    rmm::device_buffer results{total_row_groups, stream, cudf::get_current_device_resource_ref()};
    cudf::device_span<bool> results_span{static_cast<bool*>(results.data()), total_row_groups};

    // Query literal in bloom filters from each column chunk (row group).
    thrust::tabulate(
      rmm::exec_policy_nosync(stream),
      results_span.begin(),
      results_span.end(),
      [filter_span          = bloom_filter_spans.data(),
       d_scalar             = literal->get_value(),
       col_idx              = equality_col_idx,
       num_equality_columns = num_equality_columns] __device__(auto row_group_idx) {
        // Filter bitset buffer index
        auto const filter_idx  = col_idx + (num_equality_columns * row_group_idx);
        auto const filter_size = filter_span[filter_idx].size();

        // If no bloom filter, then fill in `true` as membership cannot be determined
        if (filter_size == 0) { return true; }

        // Number of filter blocks
        auto const num_filter_blocks = filter_size / bytes_per_block;

        // Create a bloom filter view.
        bloom_filter_type filter{
          reinterpret_cast<filter_block_type*>(filter_span[filter_idx].data()),
          num_filter_blocks,
          {},   // Thread scope as the same literal is being searched across different bitsets per
                // thread
          {}};  // Arrow policy with cudf::hashing::detail::XXHash_64 seeded with 0 for Arrow
                // compatibility

        // If int96_timestamp type, convert literal to string_view and query bloom
        // filter
        if constexpr (cuda::std::is_same_v<T, cudf::string_view> and
                      IS_INT96_TIMESTAMP == is_int96_timestamp::YES) {
          auto const int128_key = static_cast<__int128_t>(d_scalar.value<int64_t>());
          cudf::string_view probe_key{reinterpret_cast<char const*>(&int128_key), 12};
          return filter.contains(probe_key);
        } else {
          // Query the bloom filter and store results
          return filter.contains(d_scalar.value<T>());
        }
      });

    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                          static_cast<cudf::size_type>(total_row_groups),
                                          std::move(results),
                                          rmm::device_buffer{},
                                          0);
  }

  // Creates device columns from bloom filter membership
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type dtype,
                                           ast::literal const* const literal,
                                           rmm::cuda_stream_view stream) const
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (std::is_same_v<T, bool> or
                  (cudf::is_compound<T>() and not std::is_same_v<T, string_view>)) {
      CUDF_FAIL("Bloom filters do not support boolean or compound types");
    } else {
      // For INT96 timestamps, use cudf::string_view type and set is_int96_timestamp to YES
      if constexpr (cudf::is_timestamp<T>()) {
        if (parquet_types[equality_col_idx] == Type::INT96) {
          // For INT96 timestamps, use cudf::string_view type and set is_int96_timestamp to YES
          return query_bloom_filter<cudf::string_view, is_int96_timestamp::YES>(
            equality_col_idx, dtype, literal, stream);
        }
      }

      // For all other cases
      return query_bloom_filter<T>(equality_col_idx, dtype, literal, stream);
    }
  }
};

/**
 * @brief Collects lists of equality predicate literals in the AST expression, one list per input
 * table column. This is used in row group filtering based on bloom filters.
 */
class equality_literals_collector : public ast::detail::expression_transformer {
 public:
  equality_literals_collector() = default;

  equality_literals_collector(ast::expression const& expr, cudf::size_type num_input_columns)
    : _num_input_columns{num_input_columns}
  {
    _equality_literals.resize(_num_input_columns);
    expr.accept(*this);
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override
  {
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override
  {
    CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
                 "BloomfilterAST supports only left table");
    CUDF_EXPECTS(expr.get_column_index() < _num_input_columns,
                 "Column index cannot be more than number of columns in the table");
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override
  {
    CUDF_FAIL("Column name reference is not supported in BloomfilterAST");
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
      auto const literal_ptr = dynamic_cast<ast::literal const*>(&operands[1].get());
      CUDF_EXPECTS(literal_ptr != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      v->accept(*this);

      // Push to the corresponding column's literals list iff equality predicate is seen
      if (op == ast_operator::EQUAL) {
        auto const col_idx = v->get_column_index();
        _equality_literals[col_idx].emplace_back(const_cast<ast::literal*>(literal_ptr));
      }
    } else {
      // Just visit the operands and ignore any output
      std::ignore = visit_operands(operands);
    }

    return expr;
  }

  /**
   * @brief Vectors of equality literals in the AST expression, one per input table column
   *
   * @return Vectors of equality literals, one per input table column
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_equality_literals() &&
  {
    return std::move(_equality_literals);
  }

 private:
  std::vector<std::vector<ast::literal*>> _equality_literals;

 protected:
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
  size_type _num_input_columns;
};

/**
 * @brief Converts AST expression to bloom filter membership (BloomfilterAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class bloom_filter_expression_converter : public equality_literals_collector {
 public:
  bloom_filter_expression_converter(
    ast::expression const& expr,
    size_type num_input_columns,
    cudf::host_span<std::vector<ast::literal*> const> equality_literals)
    : _equality_literals{equality_literals}
  {
    // Set the num columns
    _num_input_columns = num_input_columns;

    // Compute and store columns literals offsets
    _col_literals_offsets.reserve(_num_input_columns + 1);
    _col_literals_offsets.emplace_back(0);

    std::transform(equality_literals.begin(),
                   equality_literals.end(),
                   std::back_inserter(_col_literals_offsets),
                   [&](auto const& col_literal_map) {
                     return _col_literals_offsets.back() +
                            static_cast<cudf::size_type>(col_literal_map.size());
                   });

    // Add this visitor
    expr.accept(*this);
  }

  /**
   * @brief Delete equality literals getter as it's not needed in the derived class
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_equality_literals() && = delete;

  // Bring all overloads of `visit` from equality_predicate_collector into scope
  using equality_literals_collector::visit;

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

      if (op == ast_operator::EQUAL) {
        // Search the literal in this input column's equality literals list and add to the offset.
        auto const col_idx            = v->get_column_index();
        auto const& equality_literals = _equality_literals[col_idx];
        auto col_literal_offset       = _col_literals_offsets[col_idx];
        auto const literal_iter       = std::find(equality_literals.cbegin(),
                                            equality_literals.cend(),
                                            dynamic_cast<ast::literal const*>(&operands[1].get()));
        CUDF_EXPECTS(literal_iter != equality_literals.end(), "Could not find the literal ptr");
        col_literal_offset += std::distance(equality_literals.cbegin(), literal_iter);

        // Evaluate boolean is_true(value) expression as NOT(NOT(value))
        auto const& value = _bloom_filter_expr.push(ast::column_reference{col_literal_offset});
        _bloom_filter_expr.push(ast::operation{
          ast_operator::NOT, _bloom_filter_expr.push(ast::operation{ast_operator::NOT, value})});
      }
      // For all other expressions, push an always true expression
      else {
        _bloom_filter_expr.push(
          ast::operation{ast_operator::NOT,
                         _bloom_filter_expr.push(ast::operation{ast_operator::NOT, _always_true})});
      }
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2) {
        _bloom_filter_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
      } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
        _bloom_filter_expr.push(ast::operation{op, new_operands.front()});
      }
    }
    return _bloom_filter_expr.back();
  }

  /**
   * @brief Returns the AST to apply on bloom filter membership.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_bloom_filter_expr() const
  {
    return _bloom_filter_expr.back();
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  cudf::host_span<std::vector<ast::literal*> const> _equality_literals;
  ast::tree _bloom_filter_expr;
  cudf::numeric_scalar<bool> _always_true_scalar{true};
  ast::literal const _always_true{_always_true_scalar};
};

/**
 * @brief Reads bloom filter data to device.
 *
 * @param sources Dataset sources
 * @param num_chunks Number of total column chunks to read
 * @param bloom_filter_data Device buffers to hold bloom filter bitsets for each chunk
 * @param bloom_filter_offsets Bloom filter offsets for all chunks
 * @param bloom_filter_sizes Bloom filter sizes for all chunks
 * @param chunk_source_map Association between each column chunk and its source
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param aligned_mr Aligned device memory resource to allocate bloom filter buffers
 */
void read_bloom_filter_data(host_span<std::unique_ptr<datasource> const> sources,
                            size_t num_chunks,
                            cudf::host_span<rmm::device_buffer> bloom_filter_data,
                            cudf::host_span<std::optional<int64_t>> bloom_filter_offsets,
                            cudf::host_span<std::optional<int32_t>> bloom_filter_sizes,
                            std::vector<size_type> const& chunk_source_map,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref aligned_mr)
{
  // Using `cuco::arrow_filter_policy` with a temporary `cuda::std::byte` key type to extract bloom
  // filter properties
  using policy_type = cuco::arrow_filter_policy<cuda::std::byte, cudf::hashing::detail::XXHash_64>;
  auto constexpr filter_block_alignment =
    alignof(cuco::bloom_filter_ref<cuda::std::byte,
                                   cuco::extent<std::size_t>,
                                   cuco::thread_scope_thread,
                                   policy_type>::filter_block_type);
  auto constexpr words_per_block = policy_type::words_per_block;

  // Read tasks for bloom filter data
  std::vector<std::future<size_t>> read_tasks;

  // Read bloom filters for all column chunks
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_chunks),
    [&](auto const chunk) {
      // If bloom filter offset absent, fill in an empty buffer and skip ahead
      if (not bloom_filter_offsets[chunk].has_value()) {
        bloom_filter_data[chunk] = {};
        return;
      }
      // Read bloom filter iff present
      auto const bloom_filter_offset = bloom_filter_offsets[chunk].value();

      // If Bloom filter size (header + bitset) is available, just read the entire thing.
      // Else just read 256 bytes which will contain the entire header and may contain the
      // entire bitset as well.
      auto constexpr bloom_filter_size_guess = 256;
      auto const initial_read_size =
        static_cast<size_t>(bloom_filter_sizes[chunk].value_or(bloom_filter_size_guess));

      // Read an initial buffer from source
      auto& source = sources[chunk_source_map[chunk]];
      auto buffer  = source->host_read(bloom_filter_offset, initial_read_size);

      // Deserialize the Bloom filter header from the buffer.
      BloomFilterHeader header;
      CompactProtocolReader cp{buffer->data(), buffer->size()};
      cp.read(&header);

      // Check if the bloom filter header is valid.
      auto const is_header_valid =
        (header.num_bytes % words_per_block) == 0 and
        header.compression.compression == BloomFilterCompression::Compression::UNCOMPRESSED and
        header.algorithm.algorithm == BloomFilterAlgorithm::Algorithm::SPLIT_BLOCK and
        header.hash.hash == BloomFilterHash::Hash::XXHASH;

      // Do not read if the bloom filter is invalid
      if (not is_header_valid) {
        bloom_filter_data[chunk] = {};
        CUDF_LOG_WARN("Encountered an invalid bloom filter header. Skipping");
        return;
      }

      // Bloom filter header size
      auto const bloom_filter_header_size = static_cast<int64_t>(cp.bytecount());
      auto const bitset_size              = static_cast<size_t>(header.num_bytes);

      // Check if we already read in the filter bitset in the initial read.
      if (initial_read_size >= bloom_filter_header_size + bitset_size) {
        bloom_filter_data[chunk] = rmm::device_buffer{
          buffer->data() + bloom_filter_header_size, bitset_size, stream, aligned_mr};
        // The allocated bloom filter buffer must be aligned
        CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(bloom_filter_data[chunk].data()) %
                         filter_block_alignment ==
                       0,
                     "Encountered misaligned bloom filter block");
      }
      // Read the bitset from datasource.
      else {
        auto const bitset_offset = bloom_filter_offset + bloom_filter_header_size;
        // Directly read to device if preferred
        if (source->is_device_read_preferred(bitset_size)) {
          bloom_filter_data[chunk] = rmm::device_buffer{bitset_size, stream, aligned_mr};
          // The allocated bloom filter buffer must be aligned
          CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(bloom_filter_data[chunk].data()) %
                           filter_block_alignment ==
                         0,
                       "Encountered misaligned bloom filter block");
          auto future_read_size =
            source->device_read_async(bitset_offset,
                                      bitset_size,
                                      static_cast<uint8_t*>(bloom_filter_data[chunk].data()),
                                      stream);

          read_tasks.emplace_back(std::move(future_read_size));
        } else {
          buffer = source->host_read(bitset_offset, bitset_size);
          bloom_filter_data[chunk] =
            rmm::device_buffer{buffer->data(), buffer->size(), stream, aligned_mr};
          // The allocated bloom filter buffer must be aligned
          CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(bloom_filter_data[chunk].data()) %
                           filter_block_alignment ==
                         0,
                       "Encountered misaligned bloom filter block");
        }
      }
    });

  // Read task sync function
  for (auto& task : read_tasks) {
    task.wait();
  }
}

}  // namespace

std::vector<rmm::device_buffer> aggregate_reader_metadata::read_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas,
  size_type total_row_groups,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref aligned_mr) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = column_schemas.size();
  auto const num_chunks        = total_row_groups * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<std::optional<int64_t>> bloom_filter_offsets(num_chunks);
  std::vector<std::optional<int32_t>> bloom_filter_sizes(num_chunks);

  // Gather all bloom filter offsets and sizes.
  size_type chunk_count = 0;

  // Flag to check if we have at least one valid bloom filter offset
  auto have_bloom_filters = false;

  // For all data sources
  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(row_group_indices.size()),
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

                        // Set `have_bloom_filters` if `bloom_filter_offset` is valid
                        if (col_meta.bloom_filter_offset.has_value()) { have_bloom_filters = true; }

                        // Map each column chunk to its source index
                        chunk_source_map[chunk_count] = src_index;
                        chunk_count++;
                      });
                  });
                });

  // Exit early if we don't have any bloom filters
  if (not have_bloom_filters) { return {}; }

  // Vector to hold bloom filter data
  std::vector<rmm::device_buffer> bloom_filter_data(num_chunks);

  // Read bloom filter data
  read_bloom_filter_data(sources,
                         num_chunks,
                         bloom_filter_data,
                         bloom_filter_offsets,
                         bloom_filter_sizes,
                         chunk_source_map,
                         stream,
                         aligned_mr);

  // Return bloom filter data
  return bloom_filter_data;
}

std::vector<Type> aggregate_reader_metadata::get_parquet_types(
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas) const
{
  std::vector<Type> parquet_types(column_schemas.size());
  // Find a source with at least one row group
  auto const src_iter = std::find_if(row_group_indices.begin(),
                                     row_group_indices.end(),
                                     [](auto const& rg) { return rg.size() > 0; });
  CUDF_EXPECTS(src_iter != row_group_indices.end(), "");

  // Source index
  auto const src_index = std::distance(row_group_indices.begin(), src_iter);
  std::transform(column_schemas.begin(),
                 column_schemas.end(),
                 parquet_types.begin(),
                 [&](auto const schema_idx) {
                   // Use the first row group in this source
                   auto constexpr row_group_index = 0;
                   return get_column_metadata(row_group_index, src_index, schema_idx).type;
                 });

  return parquet_types;
}

std::pair<std::optional<std::vector<std::vector<size_type>>>, bool>
aggregate_reader_metadata::apply_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());

  // Collect equality literals for each input table column
  auto const equality_literals =
    equality_literals_collector{filter.get(), num_input_columns}.get_equality_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> equality_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  equality_literals.begin(),
                  std::back_inserter(equality_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Return early if no column with equality predicate(s)
  if (equality_col_schemas.empty()) { return {std::nullopt, false}; }

  // Required alignment:
  // https://github.com/NVIDIA/cuCollections/blob/deab5799f3e4226cb8a49acf2199c03b14941ee4/include/cuco/detail/bloom_filter/bloom_filter_impl.cuh#L55-L67
  using policy_type = cuco::arrow_filter_policy<cuda::std::byte, cudf::hashing::detail::XXHash_64>;
  auto constexpr alignment = alignof(cuco::bloom_filter_ref<cuda::std::byte,
                                                            cuco::extent<std::size_t>,
                                                            cuco::thread_scope_thread,
                                                            policy_type>::filter_block_type);

  // Aligned resource adaptor to allocate bloom filter buffers with
  auto aligned_mr =
    rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource(), alignment);

  // Read a vector of bloom filter bitset device buffers for all columns with equality
  // predicate(s) across all row groups
  auto bloom_filter_data = read_bloom_filters(
    sources, input_row_group_indices, equality_col_schemas, total_row_groups, stream, aligned_mr);

  // No bloom filter buffers, return early
  if (bloom_filter_data.empty()) { return {std::nullopt, false}; }

  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(input_row_group_indices, equality_col_schemas);

  // Create spans from bloom filter bitset buffers to use in cuco::bloom_filter_ref.
  std::vector<cudf::device_span<cuda::std::byte>> h_bloom_filter_spans;
  h_bloom_filter_spans.reserve(bloom_filter_data.size());
  std::transform(bloom_filter_data.begin(),
                 bloom_filter_data.end(),
                 std::back_inserter(h_bloom_filter_spans),
                 [&](auto& buffer) {
                   return cudf::device_span<cuda::std::byte>{
                     static_cast<cuda::std::byte*>(buffer.data()), buffer.size()};
                 });

  // Copy bloom filter bitset spans to device
  auto const bloom_filter_spans = cudf::detail::make_device_uvector_async(
    h_bloom_filter_spans, stream, cudf::get_current_device_resource_ref());

  // Create a bloom filter query table caster
  bloom_filter_caster const bloom_filter_col{bloom_filter_spans,
                                             parquet_types,
                                             static_cast<size_t>(total_row_groups),
                                             equality_col_schemas.size()};

  // Converts bloom filter membership for equality predicate columns to a table
  // containing a column for each `col[i] == literal` predicate to be evaluated.
  // The table contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<cudf::column>> bloom_filter_membership_columns;
  size_t equality_col_idx = 0;
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(output_dtypes.size()),
    [&](auto input_col_idx) {
      auto const& dtype = output_dtypes[input_col_idx];

      // Skip if no equality literals for this column
      if (equality_literals[input_col_idx].empty()) { return; }

      // Skip if non-comparable (compound) type except string
      if (cudf::is_compound(dtype) and dtype.id() != cudf::type_id::STRING) { return; }

      // Add a column for all literals associated with an equality column
      for (auto const& literal : equality_literals[input_col_idx]) {
        bloom_filter_membership_columns.emplace_back(cudf::type_dispatcher<dispatch_storage_type>(
          dtype, bloom_filter_col, equality_col_idx, dtype, literal, stream));
      }
      equality_col_idx++;
    });

  // Create a table from columns
  auto bloom_filter_membership_table = cudf::table(std::move(bloom_filter_membership_columns));

  // Convert AST to BloomfilterAST expression with reference to bloom filter membership
  // in above `bloom_filter_membership_table`
  bloom_filter_expression_converter bloom_filter_expr{
    filter.get(), num_input_columns, {equality_literals}};

  // Filter bloom filter membership table with the BloomfilterAST expression and collect
  // filtered row group indices
  return {collect_filtered_row_group_indices(bloom_filter_membership_table,
                                             bloom_filter_expr.get_bloom_filter_expr(),
                                             input_row_group_indices,
                                             stream),
          true};
}

}  // namespace cudf::io::parquet::detail
