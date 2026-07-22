/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/unicode_normalize.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/span>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdint>

namespace nvtext {
namespace detail {
namespace {

// ---------------------------------------------------------------------------
// Composition exclusion: ~70 codepoints explicitly excluded from NFC/NFKC
// composition (Unicode 15, DerivedNormalizationProps.txt).
// ---------------------------------------------------------------------------
__device__ __constant__ uint32_t COMPOSITION_EXCLUSIONS[] = {
  0x0958u,  0x0959u,  0x095Au,  0x095Bu,  0x095Cu,  0x095Du,  0x095Eu,  0x095Fu,  0x09DCu,
  0x09DDu,  0x09DFu,  0x0A33u,  0x0A36u,  0x0A59u,  0x0A5Au,  0x0A5Bu,  0x0A5Cu,  0x0A5Eu,
  0x0B5Cu,  0x0B5Du,  0x0F43u,  0x0F4Du,  0x0F52u,  0x0F57u,  0x0F5Cu,  0x0F69u,  0x0F76u,
  0x0F78u,  0x0F80u,  0x0F93u,  0x0F9Du,  0x0FA2u,  0x0FA7u,  0x0FACu,  0x0FB9u,  0xFB1Du,
  0xFB1Fu,  0xFB2Au,  0xFB2Bu,  0xFB2Cu,  0xFB2Du,  0xFB2Eu,  0xFB2Fu,  0xFB30u,  0xFB31u,
  0xFB32u,  0xFB33u,  0xFB34u,  0xFB35u,  0xFB36u,  0xFB38u,  0xFB39u,  0xFB3Au,  0xFB3Bu,
  0xFB3Cu,  0xFB3Eu,  0xFB40u,  0xFB41u,  0xFB43u,  0xFB44u,  0xFB46u,  0xFB47u,  0xFB48u,
  0xFB49u,  0xFB4Au,  0xFB4Bu,  0xFB4Cu,  0xFB4Du,  0xFB4Eu,  0x2ADCu,  0x1D15Eu, 0x1D15Fu,
  0x1D160u, 0x1D161u, 0x1D162u, 0x1D163u, 0x1D164u, 0x1D1BBu, 0x1D1BCu, 0x1D1BDu, 0x1D1BEu,
  0x1D1BFu, 0x1D1C0u,
};
constexpr int COMPOSITION_EXCLUSIONS_SIZE =
  static_cast<int>(sizeof(COMPOSITION_EXCLUSIONS) / sizeof(COMPOSITION_EXCLUSIONS[0]));

__device__ bool is_composition_exclusion(uint32_t cp)
{
  for (int i = 0; i < COMPOSITION_EXCLUSIONS_SIZE; ++i) {
    if (COMPOSITION_EXCLUSIONS[i] == cp) return true;
  }
  return false;
}

/**
 * @brief Scatter CCC values from the CCC column into a codepoint-indexed table.
 *
 * Codepoints are pre-converted from the hex column via hex_to_integers.
 * One invocation per UnicodeData.txt row.
 */
struct scatter_ccc_fn {
  cudf::column_device_view ccc_col;              // INT32: CCC values
  cuda::std::span<uint32_t const> d_codepoints;  // pre-converted codepoints
  cuda::std::span<uint8_t> ccc_table;            // output: CCC indexed by codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    uint32_t const cp = d_codepoints[idx];
    if (cp <= MAX_CODEPOINT) {
      ccc_table[cp] = static_cast<uint8_t>(ccc_col.element<int32_t>(idx));
    }
  }
};

/**
 * @brief Invoke @p fn for each space-separated hex token in a decomp mapping string.
 *
 * Returns immediately for empty strings or, when @p apply_compat is false, for
 * compatibility mappings (strings that begin with '<').  When @p apply_compat is
 * true the leading "<tag> " prefix is consumed before iteration starts.
 */
template <typename Fn>
__device__ void for_each_decomp_token(cudf::string_view d_str, bool apply_compat, Fn fn)
{
  char const* const ptr      = d_str.data();
  cudf::size_type const size = d_str.size_bytes();
  if (size == 0) { return; }
  bool const is_compat = (ptr[0] == '<');
  if (is_compat && !apply_compat) { return; }
  cudf::size_type pos = 0;
  if (is_compat) {
    while (pos < size && ptr[pos] != '>') {
      ++pos;
    }
    pos += 2;  // skip '>' and the following space
  }
  while (pos < size) {
    while (pos < size && ptr[pos] == ' ') {
      ++pos;
    }
    cudf::size_type const tok_start = pos;
    while (pos < size && ptr[pos] != ' ') {
      ++pos;
    }
    if (pos > tok_start) { fn(ptr + tok_start, pos - tok_start); }
  }
}

/**
 * @brief Count space-separated hex tokens in a decomp mapping string.
 * One invocation per row; result written directly by thrust::transform.
 */
struct count_decomp_tokens_fn {
  cudf::column_device_view decomp_map;
  bool apply_compat;

  __device__ int32_t operator()(cudf::size_type idx) const
  {
    int32_t count = 0;
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx),
                          apply_compat,
                          [&count](char const*, cudf::size_type) { ++count; });
    return count;
  }
};

/**
 * @brief Write decomposition codepoints into the flat decomp_table.
 * One invocation per row; uses pre-computed per-codepoint offsets for placement.
 */
struct write_decomp_tokens_fn {
  cudf::column_device_view decomp_map;
  bool apply_compat;
  cuda::std::span<uint32_t const> d_codepoints;       // parsed codepoint per row
  cuda::std::span<uint32_t const> decomp_cp_offsets;  // write-start per codepoint
  cuda::std::span<uint32_t> decomp_table;             // flat output decomp codepoints

  __device__ void operator()(cudf::size_type idx) const
  {
    uint32_t const cp  = d_codepoints[idx];
    uint32_t write_pos = decomp_cp_offsets[cp];
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx),
                          apply_compat,
                          [this, &write_pos](char const* ptr, cudf::size_type size) {
                            decomp_table[write_pos++] = hex_to_cp(ptr, size);
                          });
  }
};

/**
 * @brief Build composition table entries from canonical two-token decompositions.
 * Writes a (key, value) pair per qualifying row; zero for non-qualifying rows.
 */
struct build_comp_table_fn {
  cudf::column_device_view decomp_map;
  cuda::std::span<uint32_t const> d_codepoints;  // parsed codepoint per row
  cuda::std::span<int32_t const> d_counts;       // token count per row
  cuda::std::span<uint8_t const> ccc_table;      // CCC indexed by codepoint
  cuda::std::span<uint64_t> d_comp_keys;         // output: composition key
  cuda::std::span<uint32_t> d_comp_values;       // output: composed codepoint

  __device__ void operator()(cudf::size_type idx) const
  {
    d_comp_keys[idx]   = 0;
    d_comp_values[idx] = 0;
    if (d_counts[idx] != 2) { return; }
    // apply_compat=false: skip compatibility mappings (<tag> prefix)
    uint32_t tokens[2] = {0, 0};
    int32_t tok        = 0;
    for_each_decomp_token(decomp_map.element<cudf::string_view>(idx),
                          /*apply_compat=*/false,
                          [&tokens, &tok](char const* ptr, cudf::size_type size) {
                            if (tok < 2) { tokens[tok++] = hex_to_cp(ptr, size); }
                          });
    if (tok < 2) { return; }
    auto const composed  = d_codepoints[idx];
    auto const starter   = tokens[0];
    auto const combining = tokens[1];
    if (is_composition_exclusion(composed)) { return; }
    if (starter > MAX_CODEPOINT || combining > MAX_CODEPOINT) { return; }
    if (ccc_table[starter] != 0) { return; }  // non-starter decomposition — excluded
    d_comp_keys[idx]   = (static_cast<uint64_t>(starter) << 32u) | combining;
    d_comp_values[idx] = composed;
  }
};

/**
 * @brief Transitvely decompose a single Unicode codepoint and invoke @p fn with the result.
 *
 * Runs the full NFD/NFKD ping-pong expansion loop.  @p fn is called as
 * fn(buf, count) where buf[0..count) holds the expanded codepoints.
 * Not called at all for non-lead UTF-8 bytes (idx is a byte index).
 */
template <typename Fn>
__device__ void for_each_decomposed_cp(int64_t idx,
                                       cuda::std::span<char const> chars,
                                       cuda::std::span<uint32_t const> decomp_offsets,
                                       cuda::std::span<uint32_t const> decomp_table,
                                       Fn fn)
{
  if (!cudf::strings::detail::is_begin_utf8_char(chars[idx])) { return; }
  cudf::char_utf8 ch = static_cast<unsigned char>(chars[idx]);
  if (ch > 0x7F) { cudf::strings::detail::to_char_utf8(chars.data() + idx, ch); }
  uint32_t buf_a[MAX_DECOMP_EXPAND];
  uint32_t buf_b[MAX_DECOMP_EXPAND];
  int32_t count_a = 1;
  buf_a[0]        = cudf::strings::detail::utf8_to_codepoint(ch);
  for (int32_t depth = 0; depth < MAX_DECOMP_DEPTH; ++depth) {
    int32_t count_b = 0;
    bool expanded   = false;
    for (int32_t i = 0; i < count_a; ++i) {
      auto const cp = buf_a[i];
      if (cp >= HANGUL_SBASE && cp <= HANGUL_SEND) {
        count_b += hangul_decompose(cp, buf_b + count_b);
        expanded = true;
      } else {
        auto const dstart = decomp_offsets[cp];
        auto const dend   = decomp_offsets[cp + 1u];
        if (dstart == dend) {
          buf_b[count_b++] = cp;
        } else {
          for (uint32_t j = dstart; j < dend && count_b < MAX_DECOMP_EXPAND; ++j) {
            buf_b[count_b++] = decomp_table[j];
          }
          expanded = true;
        }
      }
    }
    for (int32_t i = 0; i < count_b; ++i) {
      buf_a[i] = buf_b[i];
    }
    count_a = count_b;
    if (!expanded) { break; }
  }
  fn(buf_a, count_a);
}

/**
 * @brief Count output codepoints for the input byte at @p idx (size pass).
 * Non-lead bytes return 0.
 */
struct decompose_size_fn {
  cuda::std::span<char const> d_input_chars;
  cuda::std::span<uint32_t const> decomp_offsets;
  cuda::std::span<uint32_t const> decomp_table;

  __device__ int32_t operator()(int64_t idx) const
  {
    int32_t count = 0;
    for_each_decomposed_cp(
      idx, d_input_chars, decomp_offsets, decomp_table, [&count](uint32_t const*, int32_t n) {
        count = n;
      });
    return count;
  }
};

/**
 * @brief Write decomposed codepoints and CCCs for the input byte at @p idx (fill pass).
 * Non-lead bytes are skipped. Writes to pre-scanned positions in d_out_cps / d_out_ccc.
 */
struct decompose_fill_fn {
  cuda::std::span<char const> d_input_chars;
  cuda::std::span<uint32_t const> decomp_offsets;
  cuda::std::span<uint32_t const> decomp_table;
  cuda::std::span<uint8_t const> ccc_table;
  cuda::std::span<uint32_t const> d_out_positions;  // exclusive-scan of expanded sizes
  cuda::std::span<uint32_t> d_out_cps;
  cuda::std::span<uint8_t> d_out_ccc;

  __device__ void operator()(int64_t idx) const
  {
    for_each_decomposed_cp(idx,
                           d_input_chars,
                           decomp_offsets,
                           decomp_table,
                           [this, idx](uint32_t const* cps, int32_t count) {
                             auto const out_pos = d_out_positions[idx];
                             for (int32_t i = 0; i < count; ++i) {
                               auto const cp          = cps[i];
                               d_out_cps[out_pos + i] = cp;
                               d_out_ccc[out_pos + i] = (cp <= MAX_CODEPOINT) ? ccc_table[cp] : 0u;
                             }
                           });
  }
};

/**
 * @brief Stable-sort combining mark runs within a string's codepoint slice.
 * One invocation per string; insertion-sort each maximal run of CCC>0 marks.
 */
struct reorder_fn {
  cuda::std::span<uint32_t> d_cps;
  cuda::std::span<uint8_t> d_ccc;
  cuda::std::span<int64_t const> d_str_cp_offsets;

  __device__ void operator()(cudf::size_type str_idx) const
  {
    auto const cp_start = d_str_cp_offsets[str_idx];
    auto const cp_end   = d_str_cp_offsets[str_idx + 1];
    auto run_start      = cp_start;
    for (int64_t i = cp_start; i <= cp_end; ++i) {
      bool const is_combining = (i < cp_end) && (d_ccc[i] > 0);
      if (!is_combining) {
        auto const run_len = i - run_start;
        if (run_len > 1) {
          for (int64_t j = run_start + 1; j < i; ++j) {
            auto const cp_j  = d_cps[j];
            auto const ccc_j = d_ccc[j];
            int64_t k        = j - 1;
            while (k >= run_start && d_ccc[k] > ccc_j) {
              d_cps[k + 1] = d_cps[k];
              d_ccc[k + 1] = d_ccc[k];
              --k;
            }
            d_cps[k + 1] = cp_j;
            d_ccc[k + 1] = ccc_j;
          }
        }
        run_start = i + 1;
      }
    }
  }
};

/**
 * @brief Canonical composition pass (NFC/NFKC only).
 *
 * All threads in the block cooperatively load the composition table into
 * shared memory, then thread 0 performs the sequential composition scan.
 * Consumed codepoints are zeroed and compacted during the UTF-8 encoding pass.
 *
 * @p smem_size must be comp_table_size * (sizeof(uint64_t) + sizeof(uint32_t))
 */
CUDF_KERNEL void unicode_compose_kernel(uint32_t* d_cps,
                                        uint8_t* d_ccc,
                                        int64_t const* d_str_cp_offsets,
                                        cudf::size_type num_strings,
                                        uint64_t const* comp_keys,
                                        uint32_t const* comp_values,
                                        int64_t comp_table_size)
{
  extern __shared__ char smem[];
  auto* const s_keys   = reinterpret_cast<uint64_t*>(smem);
  auto* const s_values = reinterpret_cast<uint32_t*>(smem + comp_table_size * sizeof(uint64_t));

  // All threads cooperatively load the composition lookup table into shared memory
  for (auto i = static_cast<int64_t>(threadIdx.x); i < comp_table_size; i += blockDim.x) {
    s_keys[i]   = comp_keys[i];
    s_values[i] = comp_values[i];
  }
  __syncthreads();

  if (threadIdx.x != 0) { return; }

  auto const str_idx = static_cast<cudf::size_type>(blockIdx.x);
  if (str_idx >= num_strings) { return; }

  auto const cp_start = d_str_cp_offsets[str_idx];
  auto const cp_end   = d_str_cp_offsets[str_idx + 1];

  int64_t last_starter = -1;
  uint8_t last_class   = 0;

  for (int64_t i = cp_start; i < cp_end; ++i) {
    if (d_cps[i] == 0u) { continue; }  // already consumed
    uint8_t const ccc = d_ccc[i];
    if (ccc == 0u) {
      // New starter — first try Hangul algorithmic composition
      if (last_starter >= 0) {
        uint32_t const composed_hangul = hangul_compose(d_cps[last_starter], d_cps[i]);
        if (composed_hangul != 0u) {
          d_cps[last_starter] = composed_hangul;
          d_cps[i]            = 0u;
          d_ccc[i]            = 0u;
          continue;
        }
      }
      last_starter = i;
      last_class   = 0u;
    } else {
      // Combining mark — compose with last starter if not blocked
      if (last_starter >= 0 && last_class < ccc) {
        auto const key = (static_cast<uint64_t>(d_cps[last_starter]) << 32u) | d_cps[i];
        auto const* const it =
          thrust::lower_bound(thrust::seq, s_keys, s_keys + comp_table_size, key);
        if (it != s_keys + comp_table_size && *it == key) {
          d_cps[last_starter] = s_values[it - s_keys];
          d_cps[i]            = 0u;
          d_ccc[i]            = 0u;
          continue;
        }
      }
      last_class = ccc;
    }
  }
}

}  // anonymous namespace
}  // namespace detail

// Named struct rather than a lambda: __device__ extended lambdas are not
// permitted inside constructors.
struct is_nonzero_comp_key {
  __device__ bool operator()(uint64_t k) const { return k != 0ull; }
};

struct unicode_normalizer::unicode_normalizer_impl {
  rmm::device_uvector<uint32_t> decomp_offsets;  // size DECOMP_OFFSETS_SIZE
  rmm::device_uvector<uint32_t> decomp_table;    // flat replacement codepoints
  rmm::device_uvector<uint8_t> ccc_table;        // size CODEPOINT_TABLE_SIZE
  rmm::device_uvector<uint64_t> comp_keys;       // sorted (starter<<32|combining)
  rmm::device_uvector<uint32_t> comp_values;     // parallel composed codepoints
  unicode_normalization_form form;

  unicode_normalizer_impl(rmm::device_uvector<uint32_t>&& decomp_offsets,
                          rmm::device_uvector<uint32_t>&& decomp_table,
                          rmm::device_uvector<uint8_t>&& ccc_table,
                          rmm::device_uvector<uint64_t>&& comp_keys,
                          rmm::device_uvector<uint32_t>&& comp_values,
                          unicode_normalization_form form)
    : decomp_offsets(std::move(decomp_offsets)),
      decomp_table(std::move(decomp_table)),
      ccc_table(std::move(ccc_table)),
      comp_keys(std::move(comp_keys)),
      comp_values(std::move(comp_values)),
      form(form)
  {
  }
};

unicode_normalizer::unicode_normalizer(cudf::table_view const& unicode_data,
                                       unicode_normalization_form form,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(unicode_data.num_columns() == 3,
               "unicode_data table must have exactly 3 columns",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(0).type().id() == cudf::type_id::STRING,
               "unicode_data column[0] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(1).type().id() == cudf::type_id::INT32,
               "unicode_data column[1] must be INT32",
               std::invalid_argument);
  CUDF_EXPECTS(unicode_data.column(2).type().id() == cudf::type_id::STRING,
               "unicode_data column[2] must be STRING",
               std::invalid_argument);
  CUDF_EXPECTS(!cudf::has_nulls(unicode_data),
               "unicode_data table must not contain nulls",
               std::invalid_argument);

  cudf::size_type const num_rows = unicode_data.num_rows();

  // With 0 rows every codepoint maps to itself and CCC=0.
  // Allocate full-size tables (kernels index them by codepoint) zeroed.
  if (num_rows == 0) {
    auto policy     = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
    auto ccc_tbl    = rmm::device_uvector<uint8_t>(detail::CODEPOINT_TABLE_SIZE, stream);
    auto decomp_off = rmm::device_uvector<uint32_t>(detail::DECOMP_OFFSETS_SIZE, stream);
    thrust::fill(policy, ccc_tbl.begin(), ccc_tbl.end(), uint8_t{0});
    thrust::fill(policy, decomp_off.begin(), decomp_off.end(), uint32_t{0});
    _impl = std::make_unique<unicode_normalizer_impl>(std::move(decomp_off),
                                                      rmm::device_uvector<uint32_t>(0, stream),
                                                      std::move(ccc_tbl),
                                                      rmm::device_uvector<uint64_t>(0, stream),
                                                      rmm::device_uvector<uint32_t>(0, stream),
                                                      form);
    return;
  }

  auto codepoints_col =
    cudf::strings::detail::hex_to_integers(cudf::strings_column_view(unicode_data.column(0)),
                                           cudf::data_type{cudf::type_id::UINT32},
                                           stream,
                                           cudf::get_current_device_resource_ref());
  cuda::std::span<uint32_t const> const d_codepoints(codepoints_col->view().data<uint32_t>(),
                                                     static_cast<std::size_t>(num_rows));

  // --- column device views (col0 no longer needed) -------------------------
  auto const d_ccc_col    = cudf::column_device_view::create(unicode_data.column(1), stream);
  auto const d_decomp_map = cudf::column_device_view::create(unicode_data.column(2), stream);

  bool const apply_compat =
    (form == unicode_normalization_form::NFKD || form == unicode_normalization_form::NFKC);

  auto const policy   = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
  auto const row_iter = thrust::make_counting_iterator(cudf::size_type{0});

  // Build CCC table
  auto ccc_table = rmm::device_uvector<uint8_t>(detail::CODEPOINT_TABLE_SIZE, stream);
  thrust::fill(policy, ccc_table.begin(), ccc_table.end(), uint8_t{0});

  thrust::for_each_n(policy,
                     row_iter,
                     num_rows,
                     detail::scatter_ccc_fn{
                       *d_ccc_col, d_codepoints, {ccc_table.data(), detail::CODEPOINT_TABLE_SIZE}});

  // Count decomposition tokens per row
  auto d_counts = rmm::device_uvector<int32_t>(num_rows, stream);
  thrust::transform(policy,
                    row_iter,
                    row_iter + num_rows,
                    d_counts.begin(),
                    detail::count_decomp_tokens_fn{*d_decomp_map, apply_compat});

  // Build codepoint-indexed decomp offsets
  auto decomp_offsets_vec = rmm::device_uvector<uint32_t>(detail::DECOMP_OFFSETS_SIZE, stream);
  thrust::fill(policy, decomp_offsets_vec.begin(), decomp_offsets_vec.end(), uint32_t{0});

  // Scatter per-row token counts to the codepoint-indexed positions, then
  // exclusive-scan to get start offsets. The extra sentinel slot at
  // MAX_CODEPOINT+1 accumulates the total via the scan.
  thrust::scatter(
    policy, d_counts.begin(), d_counts.end(), d_codepoints.begin(), decomp_offsets_vec.begin());
  thrust::exclusive_scan(
    policy, decomp_offsets_vec.begin(), decomp_offsets_vec.end(), decomp_offsets_vec.begin());

  // Total decomp entries = sum of all per-row counts.
  uint32_t const total_decomp_size =
    thrust::reduce(policy, d_counts.begin(), d_counts.end(), uint32_t{0});

  // Fill decomp_table
  auto decomp_table_vec = rmm::device_uvector<uint32_t>(total_decomp_size, stream);
  thrust::for_each_n(
    policy,
    row_iter,
    num_rows,
    detail::write_decomp_tokens_fn{*d_decomp_map,
                                   apply_compat,
                                   {d_codepoints.data(), static_cast<std::size_t>(num_rows)},
                                   {decomp_offsets_vec.data(), detail::DECOMP_OFFSETS_SIZE},
                                   {decomp_table_vec.data(), total_decomp_size}});

  // Build composition table (NFC/NFKC only)
  auto comp_keys_vec   = rmm::device_uvector<uint64_t>(0, stream);
  auto comp_values_vec = rmm::device_uvector<uint32_t>(0, stream);
  if (form == unicode_normalization_form::NFC || form == unicode_normalization_form::NFKC) {
    auto d_comp_keys   = rmm::device_uvector<uint64_t>(num_rows, stream);
    auto d_comp_values = rmm::device_uvector<uint32_t>(num_rows, stream);
    thrust::fill(policy, d_comp_keys.begin(), d_comp_keys.end(), uint64_t{0});
    thrust::fill(policy, d_comp_values.begin(), d_comp_values.end(), uint32_t{0});

    thrust::for_each_n(
      policy,
      row_iter,
      num_rows,
      detail::build_comp_table_fn{*d_decomp_map,
                                  {d_codepoints.data(), static_cast<std::size_t>(num_rows)},
                                  {d_counts.data(), static_cast<std::size_t>(num_rows)},
                                  {ccc_table.data(), detail::CODEPOINT_TABLE_SIZE},
                                  {d_comp_keys.data(), static_cast<std::size_t>(num_rows)},
                                  {d_comp_values.data(), static_cast<std::size_t>(num_rows)}});

    int64_t const comp_size =
      thrust::count_if(policy, d_comp_keys.begin(), d_comp_keys.end(), is_nonzero_comp_key{});

    comp_keys_vec   = rmm::device_uvector<uint64_t>(comp_size, stream);
    comp_values_vec = rmm::device_uvector<uint32_t>(comp_size, stream);

    thrust::copy_if(
      policy, d_comp_keys.begin(), d_comp_keys.end(), comp_keys_vec.begin(), is_nonzero_comp_key{});
    // Use raw keys as stencil to copy matching values at the same positions
    thrust::copy_if(policy,
                    d_comp_values.begin(),
                    d_comp_values.end(),
                    d_comp_keys.begin(),
                    comp_values_vec.begin(),
                    is_nonzero_comp_key{});

    thrust::sort_by_key(
      policy, comp_keys_vec.begin(), comp_keys_vec.end(), comp_values_vec.begin());
  }

  _impl = std::make_unique<unicode_normalizer_impl>(std::move(decomp_offsets_vec),
                                                    std::move(decomp_table_vec),
                                                    std::move(ccc_table),
                                                    std::move(comp_keys_vec),
                                                    std::move(comp_values_vec),
                                                    form);
}

unicode_normalizer::~unicode_normalizer() {}

std::unique_ptr<unicode_normalizer> create_unicode_normalizer(cudf::table_view const& unicode_data,
                                                              unicode_normalization_form form,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<unicode_normalizer>(unicode_data, form, stream, mr);
}

namespace detail {

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto const [first_offset, last_offset] =
    cudf::strings::detail::get_first_and_last_offset(input, stream);
  int64_t const chars_size      = last_offset - first_offset;
  char const* const d_raw_chars = input.chars_begin(stream) + first_offset;

  if (chars_size == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }

  auto const& p        = *normalizer._impl;
  auto const policy    = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
  auto const byte_iter = thrust::make_counting_iterator(int64_t{0});

  // Spans into impl tables
  cuda::std::span<uint32_t const> const decomp_off_span(p.decomp_offsets.data(),
                                                        p.decomp_offsets.size());
  cuda::std::span<uint32_t const> const decomp_tbl_span(p.decomp_table.data(),
                                                        p.decomp_table.size());
  cuda::std::span<uint8_t const> const ccc_span(p.ccc_table.data(), p.ccc_table.size());
  cuda::std::span<char const> const chars_span(d_raw_chars, static_cast<std::size_t>(chars_size));

  // Decomposition: 1st pass count output codepoints per input byte
  auto d_expanded_sizes = rmm::device_uvector<int32_t>(chars_size, stream);
  thrust::fill(policy, d_expanded_sizes.begin(), d_expanded_sizes.end(), int32_t{0});

  thrust::transform(policy,
                    byte_iter,
                    byte_iter + chars_size,
                    d_expanded_sizes.begin(),
                    detail::decompose_size_fn{chars_span, decomp_off_span, decomp_tbl_span});

  // Exclusive scan to get per-byte output positions
  auto d_out_positions = rmm::device_uvector<uint32_t>(chars_size, stream);
  thrust::exclusive_scan(
    policy, d_expanded_sizes.begin(), d_expanded_sizes.end(), d_out_positions.begin());

  // Total output codepoints = sum of all per-byte expansion counts
  int64_t const total_cps =
    thrust::reduce(policy, d_expanded_sizes.begin(), d_expanded_sizes.end(), int64_t{0});

  auto d_cps = rmm::device_uvector<uint32_t>(total_cps, stream);
  auto d_ccc = rmm::device_uvector<uint8_t>(total_cps, stream);
  thrust::fill(policy, d_cps.begin(), d_cps.end(), uint32_t{0});

  // Fill codepoints and CCCs at pre-scanned positions
  thrust::for_each_n(
    policy,
    byte_iter,
    chars_size,
    detail::decompose_fill_fn{chars_span,
                              decomp_off_span,
                              decomp_tbl_span,
                              ccc_span,
                              {d_out_positions.data(), static_cast<std::size_t>(chars_size)},
                              {d_cps.data(), static_cast<std::size_t>(total_cps)},
                              {d_ccc.data(), static_cast<std::size_t>(total_cps)}});

  // Build per-string codepoint offset boundaries
  auto d_str_cp_offsets = rmm::device_uvector<int64_t>(input.size() + 1, stream);
  {
    auto const input_char_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    auto const* const d_out_pos = d_out_positions.data();
    auto const* const d_exp_sz  = d_expanded_sizes.data();
    int64_t const first         = first_offset;
    thrust::transform(
      policy,
      input_char_offsets,
      input_char_offsets + input.size() + 1,
      d_str_cp_offsets.begin(),
      cuda::proclaim_return_type<int64_t>([d_out_pos, d_exp_sz, first] __device__(int64_t off) {
        auto const local = off - first;
        if (local <= 0) { return 0L; }
        return static_cast<int64_t>(d_out_pos[local - 1]) + d_exp_sz[local - 1];
      }));
  }

  // Canonical Reorder
  thrust::for_each_n(
    policy,
    thrust::make_counting_iterator(cudf::size_type{0}),
    input.size(),
    detail::reorder_fn{{d_cps.data(), static_cast<std::size_t>(total_cps)},
                       {d_ccc.data(), static_cast<std::size_t>(total_cps)},
                       {d_str_cp_offsets.data(), static_cast<std::size_t>(input.size() + 1)}});

  // Canonical Composition (NFC/NFKC only)
  if (p.form == unicode_normalization_form::NFC || p.form == unicode_normalization_form::NFKC) {
    int64_t const comp_size = static_cast<int64_t>(p.comp_keys.size());
    if (comp_size > 0) {
      auto const smem_bytes =
        static_cast<std::size_t>(comp_size * (sizeof(uint64_t) + sizeof(uint32_t)));
      CUDF_EXPECTS(smem_bytes <= 48 * 1024,
                   "composition table exceeds shared memory budget",
                   std::overflow_error);
      constexpr int64_t block_size = 256;
      detail::unicode_compose_kernel<<<input.size(), block_size, smem_bytes, stream.value()>>>(
        d_cps.data(),
        d_ccc.data(),
        d_str_cp_offsets.data(),
        input.size(),
        p.comp_keys.data(),
        p.comp_values.data(),
        comp_size);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  }

  // UTF-8 Encode (two-pass)
  auto const cp_iter = thrust::make_counting_iterator(int64_t{0});
  // compute UTF-8 byte sizes per codepoint slot
  auto d_byte_sizes = rmm::device_uvector<int32_t>(total_cps, stream);
  thrust::transform(policy,
                    cp_iter,
                    cp_iter + total_cps,
                    d_byte_sizes.begin(),
                    cuda::proclaim_return_type<int32_t>([d_cps_ptr = d_cps.data()] __device__(
                                                          int64_t idx) -> int32_t {
                      uint32_t const cp = d_cps_ptr[idx];
                      if (cp == 0u) return 0;
                      auto const utf8 = cudf::strings::detail::codepoint_to_utf8(cp);
                      return static_cast<int32_t>(cudf::strings::detail::bytes_in_char_utf8(utf8));
                    }));

  auto d_byte_offsets = rmm::device_uvector<int64_t>(total_cps, stream);
  thrust::exclusive_scan(
    policy, d_byte_sizes.begin(), d_byte_sizes.end(), d_byte_offsets.begin(), int64_t{0});

  // Per-string output byte sizes via segmented reduce over codepoint ranges
  auto output_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  {
    auto const* const d_bs = d_byte_sizes.data();
    auto d_in              = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<cudf::size_type>([d_bs] __device__(int64_t idx) {
        return static_cast<cudf::size_type>(d_bs[idx]);
      }));
    auto const* const d_str_cp = d_str_cp_offsets.data();
    std::size_t temp_bytes     = 0;
    cub::DeviceSegmentedReduce::Sum(nullptr,
                                    temp_bytes,
                                    d_in,
                                    output_sizes.data(),
                                    input.size(),
                                    d_str_cp,
                                    d_str_cp + 1,
                                    stream.value());
    auto d_temp = rmm::device_buffer{temp_bytes, stream};
    cub::DeviceSegmentedReduce::Sum(d_temp.data(),
                                    temp_bytes,
                                    d_in,
                                    output_sizes.data(),
                                    input.size(),
                                    d_str_cp,
                                    d_str_cp + 1,
                                    stream.value());
  }

  auto [offsets_col, total_bytes_out] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  auto chars_buf = rmm::device_uvector<char>(total_bytes_out, stream, mr);

  // 2nd pass write UTF-8 bytes
  thrust::for_each_n(policy,
                     cp_iter,
                     total_cps,
                     [d_cps          = d_cps.data(),
                      d_byte_offsets = d_byte_offsets.data(),
                      chars_buf      = chars_buf.data()] __device__(int64_t idx) {
                       auto const cp = d_cps[idx];
                       if (cp == 0u) { return; }
                       auto const utf8 = cudf::strings::detail::codepoint_to_utf8(cp);
                       cudf::strings::detail::from_char_utf8(utf8, chars_buf + d_byte_offsets[idx]);
                     });

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_col),
                                   chars_buf.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> normalize_unicode(cudf::strings_column_view const& input,
                                                unicode_normalizer const& normalizer,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::normalize_unicode(input, normalizer, stream, mr);
}

}  // namespace nvtext
