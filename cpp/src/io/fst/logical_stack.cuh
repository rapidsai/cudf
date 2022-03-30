/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>

#include <cudf/utilities/error.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace io {
namespace fst {

/**
 * @brief Describes the kind of stack operation.
 */
enum class stack_op_type : int32_t {
  READ = 0,  ///< Operation reading what is currently on top of the stack
  PUSH = 1,  ///< Operation pushing a new item on top of the stack
  POP  = 2   ///< Operation popping the item currently on top of the stack
};

namespace detail {

/**
 * @brief A convenience struct that represents a stack opepration as a key-value pair, where the key
 * represents the stack's level and the value represents the stack symbol.
 *
 * @tparam KeyT The key type sufficient to cover all stack levels. Must be signed type as any
 * subsequence of stack operations must be able to be covered. E.g., consider the first 10
 * operations are all push and the last 10 operations are all pop operations, we need to be able to
 * represent a partial aggregate of the first ten items, which is '+10', just as well as a partial
 * aggregate of the last ten items, which is '-10'.
 * @tparam ValueT The value type that corresponds to the stack symbols (i.e., covers the stack
 * alphabet).
 */
template <typename KeyT, typename ValueT>
struct KeyValueOp {
  KeyT key;
  ValueT value;
};

/**
 * @brief Helper class to assist with radix sorting KeyValueOp instances by key.
 *
 * @tparam BYTE_SIZE The size of the KeyValueOp.
 */
template <std::size_t BYTE_SIZE>
struct KeyValueOpToUnsigned {
};

template <>
struct KeyValueOpToUnsigned<1U> {
  using UnsignedT = uint8_t;
};

template <>
struct KeyValueOpToUnsigned<2U> {
  using UnsignedT = uint16_t;
};

template <>
struct KeyValueOpToUnsigned<4U> {
  using UnsignedT = uint32_t;
};

template <>
struct KeyValueOpToUnsigned<8U> {
  using UnsignedT = uint64_t;
};

/**
 * @brief Alias template to retrieve an unsigned bit-representation that can be used for radix
 * sorting the key of a KeyValueOp.
 *
 * @tparam KeyValueOpT The KeyValueOp class template instance for which to get an unsigned
 * bit-representation
 */
template <typename KeyValueOpT>
using UnsignedKeyValueOpType = typename KeyValueOpToUnsigned<sizeof(KeyValueOpT)>::UnsignedT;

/**
 * @brief Function object class template used for converting a stack operation to a key-value store
 * operation, where the key corresponds to the stack level being accessed.
 *
 * @tparam KeyValueOpT
 * @tparam StackSymbolToStackOpTypeT
 */
template <typename KeyValueOpT, typename StackSymbolToStackOpTypeT>
struct StackSymbolToKVOp {
  template <typename StackSymbolT>
  constexpr CUDF_HOST_DEVICE KeyValueOpT operator()(StackSymbolT const& stack_symbol) const
  {
    stack_op_type stack_op = symbol_to_stack_op_type(stack_symbol);
    // PUSH => +1, POP => -1, READ => 0
    int32_t level_delta = stack_op == stack_op_type::PUSH  ? 1
                          : stack_op == stack_op_type::POP ? -1
                                                           : 0;
    return KeyValueOpT{static_cast<decltype(KeyValueOpT::key)>(level_delta), stack_symbol};
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Binary reduction operator to compute the absolute stack level from relative stack levels
 * (i.e., +1 for a PUSH, -1 for a POP operation).
 */
struct AddStackLevelFromKVOp {
  template <typename KeyT, typename ValueT>
  constexpr CUDF_HOST_DEVICE KeyValueOp<KeyT, ValueT> operator()(KeyValueOp<KeyT, ValueT> const& lhs,
                                                          KeyValueOp<KeyT, ValueT> const& rhs) const
  {
    KeyT new_level = lhs.key + rhs.key;
    return KeyValueOp<KeyT, ValueT>{new_level, rhs.value};
  }
};

/**
 * @brief Binary reduction operator that propagates a write operation for a specific key to all
 * reads of that same key. That is, if the key of LHS compares equal to the key of the RHS and if
 * the RHS is a read and the LHS is a write operation type, then we return LHS, otherwise we return
 * the RHS.
 */
template <typename StackSymbolToStackOpTypeT>
struct PopulatePopWithPush {
  template <typename KeyT, typename ValueT>
  constexpr CUDF_HOST_DEVICE KeyValueOp<KeyT, ValueT> operator()(KeyValueOp<KeyT, ValueT> const& lhs,
                                                          KeyValueOp<KeyT, ValueT> const& rhs) const
  {
    // If RHS is a read, then we need to figure out whether we can propagate the value from the LHS
    bool is_rhs_read = symbol_to_stack_op_type(rhs.value) != stack_op_type::PUSH;

    // Whether LHS is a matching write (i.e., the push operation that is on top of the stack for the
    // RHS's read)
    bool is_lhs_matching_write =
      (lhs.key == rhs.key) && symbol_to_stack_op_type(lhs.value) == stack_op_type::PUSH;

    return (is_rhs_read && is_lhs_matching_write) ? lhs : rhs;
  }

  /// Function object returning a stack operation type for a given stack symbol
  StackSymbolToStackOpTypeT symbol_to_stack_op_type;
};

/**
 * @brief Binary reduction operator that is used to replace each read_symbol occurance with the last
 * non-read_symbol that precedes such read_symbol.
 */
template <typename StackSymbolT>
struct PropagateLastWrite {
  constexpr CUDF_HOST_DEVICE StackSymbolT operator()(StackSymbolT const& lhs,
                                              StackSymbolT const& rhs) const
  {
    // If RHS is a yet-to-be-propagated, then we need to check whether we can use the LHS to fill
    bool is_rhs_read = (rhs == read_symbol);

    // We propagate the write from the LHS if it's a write
    bool is_lhs_write = (lhs != read_symbol);

    return (is_rhs_read && is_lhs_write) ? lhs : rhs;
  }

  /// The read_symbol that is supposed to be replaced
  StackSymbolT read_symbol;
};

/**
 * @brief Helper function object class to convert a KeyValueOp to the stack symbol of that
 * KeyValueOp.
 */
struct KVOpToStackSymbol {
  template <typename KeyT, typename ValueT>
  constexpr CUDF_HOST_DEVICE ValueT operator()(KeyValueOp<KeyT, ValueT> const& kv_op) const
  {
    return kv_op.value;
  }
};

/**
 * @brief Replaces all operations that apply to stack level '0' with the empty stack symbol
 */
template <typename KeyValueOpT>
struct RemapEmptyStack {
  constexpr CUDF_HOST_DEVICE KeyValueOpT operator()(KeyValueOpT const& kv_op) const
  {
    return kv_op.key == 0 ? empty_stack_symbol : kv_op;
  }
  KeyValueOpT empty_stack_symbol;
};

}  // namespace detail

/**
 * @brief Takes a sparse representation of a sequence of stack operations that either push something
 * onto the stack or pop something from the stack and resolves the symbol that is on top of the
 * stack.
 *
 * @tparam StackLevelT Signed integer type that must be sufficient to cover [-max_stack_level,
 * max_stack_level] for the given sequence of stack operations. Must be signed as it needs to cover
 * the stack level of any arbitrary subsequence of stack operations.
 * @tparam StackSymbolItT An input iterator type that provides the sequence of symbols that
 * represent stack operations
 * @tparam SymbolPositionT The index that this stack operation is supposed to apply to
 * @tparam StackSymbolToStackOpT Function object class to transform items from StackSymbolItT to
 * stack_op_type
 * @tparam TopOfStackOutItT Output iterator type to which StackSymbolT are being assigned
 * @tparam StackSymbolT The internal type being used (usually corresponding to StackSymbolItT's
 * value_type)
 * @tparam OffsetT Signed or unsigned integer type large enough to index into both the sparse input
 * sequence and the top-of-stack output sequence
 * @param[in] d_symbols Sequence of symbols that represent stack operations. Memory may alias with
 * \p d_top_of_stack
 * @param[in,out] d_symbol_positions Sequence of symbol positions (for a sparse representation),
 * sequence must be ordered in ascending order. Note, the memory of this array is repurposed for
 * double-buffering.
 * @param[in] symbol_to_stack_op Function object that returns a stack operation type (push, pop, or
 * read) for a given symbol from \p d_symbols
 * @param[out] d_top_of_stack A random access output iterator that will be populated with
 * what-is-on-top-of-the-stack for the given sequence of stack operations \p d_symbols
 * @param[in] empty_stack_symbol The symbol that will be written to top_of_stack whenever the stack
 * was empty
 * @param[in] read_symbol A symbol that may not be confused for a symbol that would push to the
 * stack
 * @param[in] num_symbols_in The number of symbols in the sparse representation
 * @param[in] num_symbols_out The number of symbols that are supposed to be filled with
 * what-is-on-top-of-the-stack
 * @param[in] stream The cuda stream to which to dispatch the work
 */
template <typename StackLevelT,
          typename StackSymbolItT,
          typename SymbolPositionT,
          typename StackSymbolToStackOpT,
          typename TopOfStackOutItT,
          typename StackSymbolT,
          typename OffsetT>
void SparseStackOpToTopOfStack(void* d_temp_storage,
                               size_t& temp_storage_bytes,
                               StackSymbolItT d_symbols,
                               SymbolPositionT* d_symbol_positions,
                               StackSymbolToStackOpT symbol_to_stack_op,
                               TopOfStackOutItT d_top_of_stack,
                               StackSymbolT empty_stack_symbol,
                               StackSymbolT read_symbol,
                               OffsetT num_symbols_in,
                               OffsetT num_symbols_out,
                               cudaStream_t stream = nullptr)
{
  // Type used to hold key-value pairs (key being the stack level and the value being the stack
  // symbol)
  using KeyValueOpT = detail::KeyValueOp<StackLevelT, StackSymbolT>;

  // The unsigned integer type that we use for radix sorting items of type KeyValueOpT
  using KVOpUnsignedT = detail::UnsignedKeyValueOpType<KeyValueOpT>;

  // Transforming sequence of stack symbols to key-value store operations, where the key corresponds
  // to the stack level of a given stack operation and the value corresponds to the stack symbol of
  // that operation
  using StackSymbolToKVOpT = detail::StackSymbolToKVOp<KeyValueOpT, StackSymbolToStackOpT>;

  // TransformInputIterator converting stack symbols to key-value store operations
  using TransformInputItT =
    cub::TransformInputIterator<KeyValueOpT, StackSymbolToKVOpT, StackSymbolItT>;

  // Converting a stack symbol that may either push or pop to a key-value store operation:
  // stack_symbol -> ([+1,0,-1], stack_symbol)
  StackSymbolToKVOpT stack_sym_to_kv_op{symbol_to_stack_op};
  TransformInputItT stack_symbols_in(d_symbols, stack_sym_to_kv_op);

  // Double-buffer for sorting along the given sequence of symbol positions (the sparse
  // representation)
  cub::DoubleBuffer<SymbolPositionT> d_symbol_positions_db{nullptr, nullptr};

  // Double-buffer for sorting the key-value store operations
  cub::DoubleBuffer<KeyValueOpT> d_kv_operations{nullptr, nullptr};

  // A double-buffer that aliases memory from d_kv_operations but offset by one item (to discard the
  // exclusive scans first item)
  cub::DoubleBuffer<KeyValueOpT> d_kv_operations_offset{nullptr, nullptr};

  // A double-buffer that aliases memory from d_kv_operations_offset with unsigned types in order to
  // be able to perform a radix sort
  cub::DoubleBuffer<KVOpUnsignedT> d_kv_operations_unsigned{nullptr, nullptr};

  constexpr std::size_t bits_per_byte = 8;
  constexpr std::size_t begin_bit     = offsetof(KeyValueOpT, key) * bits_per_byte;
  constexpr std::size_t end_bit       = begin_bit + (sizeof(KeyValueOpT::key) * bits_per_byte);

  // The key-value store operation that makes sure that reads for stack level '0' will be populated
  // with the empty_stack_symbol
  KeyValueOpT const empty_stack{0, empty_stack_symbol};

  cub::TransformInputIterator<KeyValueOpT, detail::RemapEmptyStack<KeyValueOpT>, KeyValueOpT*>
    kv_ops_scan_in(nullptr, detail::RemapEmptyStack<KeyValueOpT>{empty_stack});
  KeyValueOpT* kv_ops_scan_out = nullptr;

  //------------------------------------------------------------------------------
  // MEMORY REQUIREMENTS
  //------------------------------------------------------------------------------
  enum mem_alloc_id {
    temp_storage = 0,
    symbol_position_alt,
    kv_ops_current,
    kv_ops_alt,
    num_allocations
  };

  void* allocations[mem_alloc_id::num_allocations]            = {nullptr};
  std::size_t allocation_sizes[mem_alloc_id::num_allocations] = {0};

  std::size_t stack_level_scan_bytes      = 0;
  std::size_t stack_level_sort_bytes      = 0;
  std::size_t match_level_scan_bytes      = 0;
  std::size_t propagate_writes_scan_bytes = 0;

  // Getting temporary storage requirements for the prefix sum of the stack level after each
  // operation
  CUDA_TRY(cub::DeviceScan::InclusiveScan(nullptr,
                                          stack_level_scan_bytes,
                                          stack_symbols_in,
                                          d_kv_operations_offset.Current(),
                                          detail::AddStackLevelFromKVOp{},
                                          num_symbols_in,
                                          stream));

  // Getting temporary storage requirements for the stable radix sort (sorting by stack level of the
  // operations)
  CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                           stack_level_sort_bytes,
                                           d_kv_operations_unsigned,
                                           d_symbol_positions_db,
                                           num_symbols_in,
                                           begin_bit,
                                           end_bit,
                                           stream));

  // Getting temporary storage requirements for the scan to match pop operations with the latest
  // push of the same level
  CUDA_TRY(cub::DeviceScan::InclusiveScan(
    nullptr,
    match_level_scan_bytes,
    kv_ops_scan_in,
    kv_ops_scan_out,
    detail::PopulatePopWithPush<StackSymbolToStackOpT>{symbol_to_stack_op},
    num_symbols_in,
    stream));

  // Getting temporary storage requirements for the scan to propagate top-of-stack for spots that
  // didn't push or pop
  CUDA_TRY(cub::DeviceScan::ExclusiveScan(nullptr,
                                          propagate_writes_scan_bytes,
                                          d_top_of_stack,
                                          d_top_of_stack,
                                          detail::PropagateLastWrite<StackSymbolT>{read_symbol},
                                          empty_stack_symbol,
                                          num_symbols_out,
                                          stream));

  // Scratch memory required by the algorithms
  allocation_sizes[mem_alloc_id::temp_storage] = std::max({stack_level_scan_bytes,
                                                           stack_level_sort_bytes,
                                                           match_level_scan_bytes,
                                                           propagate_writes_scan_bytes});

  // Memory requirements by auxiliary buffers
  constexpr std::size_t extra_overlap_bytes           = 2U;
  allocation_sizes[mem_alloc_id::symbol_position_alt] = num_symbols_in * sizeof(SymbolPositionT);
  allocation_sizes[mem_alloc_id::kv_ops_current] =
    (num_symbols_in + extra_overlap_bytes) * sizeof(KeyValueOpT);
  allocation_sizes[mem_alloc_id::kv_ops_alt] =
    (num_symbols_in + extra_overlap_bytes) * sizeof(KeyValueOpT);

  // Try to alias into the user-provided temporary storage memory blob
  CUDA_TRY(cub::AliasTemporaries<mem_alloc_id::num_allocations>(
    d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));

  // If this call was just to retrieve auxiliary memory requirements or not sufficient memory was
  // provided
  if (!d_temp_storage) { return; }

  //------------------------------------------------------------------------------
  // ALGORITHM
  //------------------------------------------------------------------------------
  // Amount of temp storage available to CUB algorithms
  std::size_t cub_temp_storage_bytes = allocation_sizes[mem_alloc_id::temp_storage];

  // Temp storage for CUB algorithms
  void* d_cub_temp_storage = allocations[mem_alloc_id::temp_storage];

  // Initialize double-buffer for sorting the indexes of the sequence of sparse stack operations
  d_symbol_positions_db = cub::DoubleBuffer<SymbolPositionT>{
    d_symbol_positions,
    reinterpret_cast<SymbolPositionT*>(allocations[mem_alloc_id::symbol_position_alt])};

  // Initialize double-buffer for sorting the indexes of the sequence of sparse stack operations
  d_kv_operations = cub::DoubleBuffer<KeyValueOpT>{
    reinterpret_cast<KeyValueOpT*>(allocations[mem_alloc_id::kv_ops_current]),
    reinterpret_cast<KeyValueOpT*>(allocations[mem_alloc_id::kv_ops_alt])};

  d_kv_operations_offset =
    cub::DoubleBuffer<KeyValueOpT>{d_kv_operations.Current(), d_kv_operations.Alternate()};

  // Compute prefix sum of the stack level after each operation
  CUDA_TRY(cub::DeviceScan::InclusiveScan(d_cub_temp_storage,
                                          cub_temp_storage_bytes,
                                          stack_symbols_in,
                                          d_kv_operations_offset.Current(),
                                          detail::AddStackLevelFromKVOp{},
                                          num_symbols_in,
                                          stream));

  // Stable radix sort, sorting by stack level of the operations
  d_kv_operations_unsigned = cub::DoubleBuffer<KVOpUnsignedT>{
    reinterpret_cast<KVOpUnsignedT*>(d_kv_operations_offset.Current()),
    reinterpret_cast<KVOpUnsignedT*>(d_kv_operations_offset.Alternate())};
  CUDA_TRY(cub::DeviceRadixSort::SortPairs(d_cub_temp_storage,
                                           cub_temp_storage_bytes,
                                           d_kv_operations_unsigned,
                                           d_symbol_positions_db,
                                           num_symbols_in,
                                           begin_bit,
                                           end_bit,
                                           stream));

  // TransformInputIterator that remaps all operations on stack level 0 to the empty stack symbol
  kv_ops_scan_in  = {reinterpret_cast<KeyValueOpT*>(d_kv_operations_unsigned.Current()),
                    detail::RemapEmptyStack<KeyValueOpT>{empty_stack}};
  kv_ops_scan_out = reinterpret_cast<KeyValueOpT*>(d_kv_operations_unsigned.Alternate());

  // Exclusive scan to match pop operations with the latest push operation of that level
  CUDA_TRY(cub::DeviceScan::InclusiveScan(
    d_cub_temp_storage,
    cub_temp_storage_bytes,
    kv_ops_scan_in,
    kv_ops_scan_out,
    detail::PopulatePopWithPush<StackSymbolToStackOpT>{symbol_to_stack_op},
    num_symbols_in,
    stream));

  // Fill the output tape with read-symbol
  thrust::fill(thrust::cuda::par.on(stream),
               thrust::device_ptr<StackSymbolT>{d_top_of_stack},
               thrust::device_ptr<StackSymbolT>{d_top_of_stack + num_symbols_out},
               read_symbol);

  // Transform the key-value operations to the stack symbol they represent
  cub::TransformInputIterator<StackSymbolT, detail::KVOpToStackSymbol, KeyValueOpT*>
    kv_op_to_stack_sym_it(kv_ops_scan_out, detail::KVOpToStackSymbol{});

  // Scatter the stack symbols to the output tape (spots that are not scattered to have been
  // pre-filled with the read-symbol)
  thrust::scatter(thrust::cuda::par.on(stream),
                  kv_op_to_stack_sym_it,
                  kv_op_to_stack_sym_it + num_symbols_in,
                  d_symbol_positions_db.Current(),
                  d_top_of_stack);

  // We perform an exclusive scan in order to fill the items at the very left that may
  // be reading the empty stack before there's the first push occurance in the sequence.
  // Also, we're interested in the top-of-the-stack symbol before the operation was applied.
  CUDA_TRY(cub::DeviceScan::ExclusiveScan(d_cub_temp_storage,
                                          cub_temp_storage_bytes,
                                          d_top_of_stack,
                                          d_top_of_stack,
                                          detail::PropagateLastWrite<StackSymbolT>{read_symbol},
                                          empty_stack_symbol,
                                          num_symbols_out,
                                          stream));
}

}  // namespace fst
}  // namespace io
}  // namespace cudf
