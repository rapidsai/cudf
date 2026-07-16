/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Builders for single-page Parquet files with DELTA-family encodings, used to test mini-block
// sizes no stock writer emits (the cudf and parquet-mr writers put 32 values in a mini-block,
// pyarrow and arrow-rs at most 64, while the format allows any multiple of 32). Callers pass
// the column values plus the DELTA block geometry (block_size, mini_block_count) and get back
// the complete file bytes.

// ---------------------------------------------------------------------------------------------
// minimal thrift compact-protocol writer (just what the page header and footer need)
// ---------------------------------------------------------------------------------------------
struct thrift_compact_writer {
  static constexpr uint8_t t_bool_true  = 1;
  static constexpr uint8_t t_bool_false = 2;
  static constexpr uint8_t t_i32        = 5;
  static constexpr uint8_t t_i64        = 6;
  static constexpr uint8_t t_binary     = 8;
  static constexpr uint8_t t_list       = 9;
  static constexpr uint8_t t_struct     = 12;

  std::vector<uint8_t> buf;

  void uvarint(uint64_t v)
  {
    while (true) {
      uint8_t const b = v & 0x7f;
      v >>= 7;
      if (v) {
        buf.push_back(b | 0x80);
      } else {
        buf.push_back(b);
        return;
      }
    }
  }

  void zigzag(int64_t v)
  {
    uvarint((static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63));
  }

  int field(int prev_id, int field_id, uint8_t type_id)
  {
    int const delta = field_id - prev_id;
    if (delta >= 1 && delta <= 15) {
      buf.push_back(static_cast<uint8_t>((delta << 4) | type_id));
    } else {
      buf.push_back(type_id);
      zigzag(field_id);
    }
    return field_id;
  }

  int i32(int prev_id, int field_id, int64_t val)
  {
    int const p = field(prev_id, field_id, t_i32);
    zigzag(val);
    return p;
  }

  int i64(int prev_id, int field_id, int64_t val)
  {
    int const p = field(prev_id, field_id, t_i64);
    zigzag(val);
    return p;
  }

  int boolean(int prev_id, int field_id, bool val)
  {
    return field(prev_id, field_id, val ? t_bool_true : t_bool_false);
  }

  int binary(int prev_id, int field_id, std::string_view data)
  {
    int const p = field(prev_id, field_id, t_binary);
    raw_binary(data);
    return p;
  }

  void list_header(int size, uint8_t elem_type)
  {
    if (size < 15) {
      buf.push_back(static_cast<uint8_t>((size << 4) | elem_type));
    } else {
      buf.push_back(0xf0 | elem_type);
      uvarint(size);
    }
  }

  void stop() { buf.push_back(0); }
  void raw_i32(int64_t val) { zigzag(val); }
  void raw_binary(std::string_view data)
  {
    uvarint(data.size());
    buf.insert(buf.end(), data.begin(), data.end());
  }
};

// parquet format constants used by the builders
namespace parquet_delta_test {
constexpr int type_int64             = 2;
constexpr int type_byte_array        = 6;
constexpr int rep_optional           = 1;
constexpr int rep_repeated           = 2;
constexpr int converted_utf8         = 0;
constexpr int converted_list         = 3;
constexpr int enc_rle                = 3;
constexpr int enc_delta_binary       = 5;
constexpr int enc_delta_length_ba    = 6;
constexpr int enc_delta_ba           = 7;
constexpr int page_type_data_page    = 0;
constexpr int page_type_data_page_v2 = 3;
}  // namespace parquet_delta_test

// ---------------------------------------------------------------------------------------------
// DELTA_BINARY_PACKED stream encoder
// ---------------------------------------------------------------------------------------------

// pack values (padded with 0 up to `count`) at `width` bits each, LSB-first, consecutively --
// the same layout the RLE/bit-packing hybrid and the delta mini-blocks use
inline void bitpack_into(std::vector<uint8_t>& out,
                         std::vector<uint64_t> const& vals,
                         int width,
                         int count)
{
  size_t const base = out.size();
  out.resize(base + static_cast<size_t>(count) * width / 8, 0);
  size_t pos = 0;
  for (auto const v : vals) {
    for (int b = 0; b < width; b++) {
      if ((v >> b) & 1) { out[base + pos / 8] |= 1 << (pos % 8); }
      pos++;
    }
  }
}

// complete DELTA_BINARY_PACKED stream: header (block_size, mini_block_count, value count, first
// value), then per block a zigzag min-delta, one bit-width byte per mini-block, and the
// bit-packed deltas
inline std::vector<uint8_t> encode_delta_binary_packed(std::vector<int64_t> const& values,
                                                       int block_size,
                                                       int mini_block_count)
{
  assert(block_size % mini_block_count == 0 && (block_size / mini_block_count) % 32 == 0);
  std::vector<uint8_t> out;
  auto uleb = [&out](uint64_t v) {
    while (true) {
      uint8_t const b = v & 0x7f;
      v >>= 7;
      if (v) {
        out.push_back(b | 0x80);
      } else {
        out.push_back(b);
        return;
      }
    }
  };
  auto zz = [&uleb](int64_t v) {
    uleb((static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63));
  };

  uleb(block_size);
  uleb(mini_block_count);
  uleb(values.size());  // total value count, including the first value below
  zz(values.empty() ? 0 : values.front());
  if (values.size() <= 1) { return out; }

  std::vector<int64_t> deltas(values.size() - 1);
  for (size_t i = 0; i + 1 < values.size(); i++) {
    deltas[i] = values[i + 1] - values[i];
  }

  int const vpm = block_size / mini_block_count;
  for (size_t bstart = 0; bstart < deltas.size(); bstart += block_size) {
    auto const bend      = std::min(bstart + block_size, deltas.size());
    auto const min_delta = *std::min_element(deltas.begin() + bstart, deltas.begin() + bend);
    zz(min_delta);

    // per mini-block bit widths, then the packed deltas (empty trailing mini-blocks get width 0
    // and no data)
    std::vector<int> widths(mini_block_count, 0);
    std::vector<std::vector<uint64_t>> rel(mini_block_count);
    for (int m = 0; m < mini_block_count; m++) {
      auto const mstart = bstart + static_cast<size_t>(m) * vpm;
      auto const mend   = std::min(mstart + vpm, bend);
      for (size_t i = mstart; i < mend; i++) {
        auto const r = static_cast<uint64_t>(deltas[i] - min_delta);  // >= 0 by construction
        rel[m].push_back(r);
        int w = 0;
        while (r >> w) {
          w++;
        }
        widths[m] = std::max(widths[m], w);
      }
    }
    for (auto const w : widths) {
      out.push_back(static_cast<uint8_t>(w));
    }
    for (int m = 0; m < mini_block_count; m++) {
      if (!rel[m].empty()) { bitpack_into(out, rel[m], widths[m], vpm); }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// single-page file assembly
// ---------------------------------------------------------------------------------------------

// V1 data page + footer around `body` for a single REQUIRED flat column "a"
inline std::vector<uint8_t> wrap_single_page_parquet(
  std::vector<uint8_t> const& body, int num_values, int physical_type, int encoding, bool utf8)
{
  namespace pq = parquet_delta_test;

  thrift_compact_writer ph;
  int p = ph.i32(0, 1, pq::page_type_data_page);
  p     = ph.i32(p, 2, body.size());                        // uncompressed_page_size
  p     = ph.i32(p, 3, body.size());                        // compressed_page_size
  p     = ph.field(p, 5, thrift_compact_writer::t_struct);  // data_page_header
  int d = ph.i32(0, 1, num_values);
  d     = ph.i32(d, 2, encoding);
  d     = ph.i32(d, 3, pq::enc_rle);  // definition level encoding (no levels: REQUIRED)
  d     = ph.i32(d, 4, pq::enc_rle);  // repetition level encoding
  ph.stop();
  ph.stop();

  int const data_page_offset = 4;
  auto const chunk_size      = static_cast<int64_t>(ph.buf.size() + body.size());

  thrift_compact_writer fm;
  int f = fm.i32(0, 1, 1);                                // version
  f     = fm.field(f, 2, thrift_compact_writer::t_list);  // schema
  fm.list_header(2, thrift_compact_writer::t_struct);
  {  // root: group "schema" with one child
    int r = fm.binary(0, 4, "schema");
    r     = fm.i32(r, 5, 1);
    fm.stop();
  }
  {  // required column "a"
    int c = fm.i32(0, 1, physical_type);
    c     = fm.i32(c, 3, 0);  // repetition_type REQUIRED
    c     = fm.binary(c, 4, "a");
    if (utf8) { c = fm.i32(c, 6, pq::converted_utf8); }
    fm.stop();
  }
  f = fm.i64(f, 3, num_values);                       // num_rows
  f = fm.field(f, 4, thrift_compact_writer::t_list);  // row_groups
  fm.list_header(1, thrift_compact_writer::t_struct);
  fm.field(0, 1, thrift_compact_writer::t_list);  // RowGroup.columns
  fm.list_header(1, thrift_compact_writer::t_struct);
  int cc = fm.i64(0, 2, data_page_offset);                    // ColumnChunk.file_offset
  cc     = fm.field(cc, 3, thrift_compact_writer::t_struct);  // ColumnChunk.meta_data
  {
    int cm = fm.i32(0, 1, physical_type);
    cm     = fm.field(cm, 2, thrift_compact_writer::t_list);  // encodings
    fm.list_header(2, thrift_compact_writer::t_i32);
    fm.raw_i32(pq::enc_rle);
    fm.raw_i32(encoding);
    cm = fm.field(cm, 3, thrift_compact_writer::t_list);  // path_in_schema
    fm.list_header(1, thrift_compact_writer::t_binary);
    fm.raw_binary("a");
    cm = fm.i32(cm, 4, 0);  // codec UNCOMPRESSED
    cm = fm.i64(cm, 5, num_values);
    cm = fm.i64(cm, 6, chunk_size);  // total_uncompressed_size
    cm = fm.i64(cm, 7, chunk_size);  // total_compressed_size
    cm = fm.i64(cm, 9, data_page_offset);
    fm.stop();
  }
  fm.stop();                 // ColumnChunk
  fm.i64(1, 2, chunk_size);  // RowGroup.total_byte_size
  fm.i64(2, 3, num_values);  // RowGroup.num_rows
  fm.stop();                 // RowGroup
  fm.stop();                 // FileMetaData

  std::vector<uint8_t> out;
  auto append = [&out](auto const& bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  append(ph.buf);
  append(body);
  append(fm.buf);
  auto const flen = static_cast<uint32_t>(fm.buf.size());
  for (int i = 0; i < 4; i++) {
    out.push_back((flen >> (8 * i)) & 0xff);
  }
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  return out;
}

// complete file: one DELTA_BINARY_PACKED INT64 column "a"
inline std::vector<uint8_t> build_delta_binary_parquet(std::vector<int64_t> const& values,
                                                       int block_size,
                                                       int mini_block_count)
{
  auto const body = encode_delta_binary_packed(values, block_size, mini_block_count);
  return wrap_single_page_parquet(body,
                                  values.size(),
                                  parquet_delta_test::type_int64,
                                  parquet_delta_test::enc_delta_binary,
                                  false);
}

// ---------------------------------------------------------------------------------------------
// deterministic test data (self-contained splitmix64 so results never vary across platforms)
// ---------------------------------------------------------------------------------------------
inline uint64_t delta_test_rand(uint64_t& state)
{
  state += 0x9e3779b97f4a7c15ull;
  uint64_t z = state;
  z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// values whose deltas vary within [-2, regime_max], with the magnitude regime switching every 64
// values so consecutive mini-blocks get different, non-zero bit widths
inline std::vector<int64_t> delta_test_int64_values(int n, uint64_t seed = 101)
{
  constexpr int64_t regime_max[] = {5, 220, 3000, 60000};
  std::vector<int64_t> out(n);
  int64_t v = 0;
  for (int i = 0; i < n; i++) {
    out[i]        = v;
    auto const hi = regime_max[(i / 64) % 4];
    v += static_cast<int64_t>(delta_test_rand(seed) % static_cast<uint64_t>(hi + 3)) - 2;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// string encodings
// ---------------------------------------------------------------------------------------------

// alphanumeric strings with lengths varying in [1, 20]; with shared_prefixes, each string keeps
// a random-length prefix of its predecessor so the DELTA_BYTE_ARRAY prefix-length stream also
// has varying non-zero deltas
inline std::vector<std::string> delta_test_strings(int n, bool shared_prefixes, uint64_t seed = 201)
{
  constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::vector<std::string> out;
  out.reserve(n);
  std::string prev;
  for (int i = 0; i < n; i++) {
    auto const length = 1 + static_cast<size_t>(delta_test_rand(seed) % 20);
    std::string s;
    if (shared_prefixes && !prev.empty()) {
      auto const keep = delta_test_rand(seed) % (std::min(prev.size(), length) + 1);
      s               = prev.substr(0, keep);
    }
    while (s.size() < length) {
      s += alphabet[delta_test_rand(seed) % (sizeof(alphabet) - 1)];
    }
    out.push_back(s);
    prev = std::move(s);
  }
  return out;
}

// complete file: one DELTA_LENGTH_BYTE_ARRAY string column "a" (delta-encoded lengths followed
// by the concatenated string bytes)
inline std::vector<uint8_t> build_delta_length_byte_array_parquet(
  std::vector<std::string> const& strings, int block_size, int mini_block_count)
{
  std::vector<int64_t> lengths(strings.size());
  std::transform(strings.begin(), strings.end(), lengths.begin(), [](auto const& s) {
    return static_cast<int64_t>(s.size());
  });
  auto body = encode_delta_binary_packed(lengths, block_size, mini_block_count);
  for (auto const& s : strings) {
    body.insert(body.end(), s.begin(), s.end());
  }
  return wrap_single_page_parquet(body,
                                  strings.size(),
                                  parquet_delta_test::type_byte_array,
                                  parquet_delta_test::enc_delta_length_ba,
                                  true);
}

// complete file: one DELTA_BYTE_ARRAY string column "a" (front compression: delta-encoded
// shared-prefix lengths, then delta-encoded suffix lengths, then the concatenated suffixes)
inline std::vector<uint8_t> build_delta_byte_array_parquet(std::vector<std::string> const& strings,
                                                           int block_size,
                                                           int mini_block_count)
{
  std::vector<int64_t> prefix_lens, suffix_lens;
  std::string suffix_bytes;
  std::string prev;
  for (auto const& s : strings) {
    size_t lcp     = 0;
    auto const end = std::min(prev.size(), s.size());
    while (lcp < end && prev[lcp] == s[lcp]) {
      lcp++;
    }
    prefix_lens.push_back(lcp);
    suffix_lens.push_back(s.size() - lcp);
    suffix_bytes.append(s, lcp, std::string::npos);
    prev = s;
  }
  auto body                = encode_delta_binary_packed(prefix_lens, block_size, mini_block_count);
  auto const suffix_stream = encode_delta_binary_packed(suffix_lens, block_size, mini_block_count);
  body.insert(body.end(), suffix_stream.begin(), suffix_stream.end());
  body.insert(body.end(), suffix_bytes.begin(), suffix_bytes.end());
  return wrap_single_page_parquet(body,
                                  strings.size(),
                                  parquet_delta_test::type_byte_array,
                                  parquet_delta_test::enc_delta_ba,
                                  true);
}

// ---------------------------------------------------------------------------------------------
// LIST<INT64>: one optional list column "col" of optional int64 "element" (max_def_level 3,
// max_rep_level 1), no null lists or elements -- empty lists only. Emitted as a single
// uncompressed V2 data page whose rep/def levels are RLE/bit-packed hybrid runs.
// ---------------------------------------------------------------------------------------------

// encode `levels` at `width` bits as one bit-packed hybrid run (padded to a multiple of 8)
inline std::vector<uint8_t> encode_levels_bit_packed(std::vector<int> const& levels, int width)
{
  auto const groups = (levels.size() + 7) / 8;
  std::vector<uint8_t> out;
  auto header = static_cast<uint64_t>((groups << 1) | 1);
  while (true) {
    uint8_t const b = header & 0x7f;
    header >>= 7;
    if (header) {
      out.push_back(b | 0x80);
    } else {
      out.push_back(b);
      break;
    }
  }
  std::vector<uint64_t> vals(levels.begin(), levels.end());
  bitpack_into(out, vals, width, groups * 8);
  return out;
}

inline std::vector<uint8_t> build_delta_binary_list_parquet(
  std::vector<std::vector<int64_t>> const& lists, int block_size, int mini_block_count)
{
  namespace pq = parquet_delta_test;

  std::vector<int64_t> leaf_values;
  std::vector<int> rep_levels, def_levels;
  int num_nulls = 0;
  for (auto const& list : lists) {
    if (list.empty()) {  // empty list: one level entry, def < max_def, no leaf value
      rep_levels.push_back(0);
      def_levels.push_back(1);
      num_nulls++;
      continue;
    }
    for (size_t j = 0; j < list.size(); j++) {
      rep_levels.push_back(j == 0 ? 0 : 1);
      def_levels.push_back(3);
      leaf_values.push_back(list[j]);
    }
  }
  auto const num_values = static_cast<int>(rep_levels.size());
  auto const num_rows   = static_cast<int>(lists.size());

  auto const rep       = encode_levels_bit_packed(rep_levels, 1);  // max_rep_level 1
  auto const dfn       = encode_levels_bit_packed(def_levels, 2);  // max_def_level 3
  auto const values    = encode_delta_binary_packed(leaf_values, block_size, mini_block_count);
  auto const page_size = static_cast<int64_t>(rep.size() + dfn.size() + values.size());

  thrift_compact_writer ph;
  int p = ph.i32(0, 1, pq::page_type_data_page_v2);
  p     = ph.i32(p, 2, page_size);
  p     = ph.i32(p, 3, page_size);
  p     = ph.field(p, 8, thrift_compact_writer::t_struct);  // data_page_header_v2
  int d = ph.i32(0, 1, num_values);
  d     = ph.i32(d, 2, num_nulls);
  d     = ph.i32(d, 3, num_rows);
  d     = ph.i32(d, 4, pq::enc_delta_binary);
  d     = ph.i32(d, 5, dfn.size());  // definition_levels_byte_length
  d     = ph.i32(d, 6, rep.size());  // repetition_levels_byte_length
  d     = ph.boolean(d, 7, false);   // is_compressed
  ph.stop();
  ph.stop();

  int const data_page_offset = 4;
  auto const chunk_size      = static_cast<int64_t>(ph.buf.size()) + page_size;

  thrift_compact_writer fm;
  int f = fm.i32(0, 1, 2);                                // version
  f     = fm.field(f, 2, thrift_compact_writer::t_list);  // schema
  fm.list_header(4, thrift_compact_writer::t_struct);
  {  // root: group "schema" with one child
    int r = fm.binary(0, 4, "schema");
    r     = fm.i32(r, 5, 1);
    fm.stop();
  }
  {  // optional group "col", converted/logical type LIST, one child
    int c = fm.i32(0, 3, pq::rep_optional);
    c     = fm.binary(c, 4, "col");
    c     = fm.i32(c, 5, 1);
    c     = fm.i32(c, 6, pq::converted_list);
    c     = fm.field(c, 10, thrift_compact_writer::t_struct);  // logicalType
    fm.field(0, 3, thrift_compact_writer::t_struct);           //   LIST (empty struct)
    fm.stop();
    fm.stop();
    fm.stop();
  }
  {  // repeated group "list" with one child
    int g = fm.i32(0, 3, pq::rep_repeated);
    g     = fm.binary(g, 4, "list");
    g     = fm.i32(g, 5, 1);
    fm.stop();
  }
  {  // optional int64 "element"
    int e = fm.i32(0, 1, pq::type_int64);
    e     = fm.i32(e, 3, pq::rep_optional);
    e     = fm.binary(e, 4, "element");
    fm.stop();
  }
  f = fm.i64(f, 3, num_rows);
  f = fm.field(f, 4, thrift_compact_writer::t_list);  // row_groups
  fm.list_header(1, thrift_compact_writer::t_struct);
  fm.field(0, 1, thrift_compact_writer::t_list);  // RowGroup.columns
  fm.list_header(1, thrift_compact_writer::t_struct);
  int cc = fm.i64(0, 2, data_page_offset);
  cc     = fm.field(cc, 3, thrift_compact_writer::t_struct);  // meta_data
  {
    int cm = fm.i32(0, 1, pq::type_int64);
    cm     = fm.field(cm, 2, thrift_compact_writer::t_list);  // encodings
    fm.list_header(2, thrift_compact_writer::t_i32);
    fm.raw_i32(pq::enc_rle);
    fm.raw_i32(pq::enc_delta_binary);
    cm = fm.field(cm, 3, thrift_compact_writer::t_list);  // path_in_schema
    fm.list_header(3, thrift_compact_writer::t_binary);
    fm.raw_binary("col");
    fm.raw_binary("list");
    fm.raw_binary("element");
    cm = fm.i32(cm, 4, 0);           // codec UNCOMPRESSED
    cm = fm.i64(cm, 5, num_values);  // num_values counts level entries incl. empties
    cm = fm.i64(cm, 6, chunk_size);
    cm = fm.i64(cm, 7, chunk_size);
    cm = fm.i64(cm, 9, data_page_offset);
    fm.stop();
  }
  fm.stop();                 // ColumnChunk
  fm.i64(1, 2, chunk_size);  // RowGroup.total_byte_size
  fm.i64(2, 3, num_rows);    // RowGroup.num_rows
  fm.stop();                 // RowGroup
  fm.stop();                 // FileMetaData

  std::vector<uint8_t> out;
  auto append = [&out](auto const& bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  append(ph.buf);
  append(rep);
  append(dfn);
  append(values);
  append(fm.buf);
  auto const flen = static_cast<uint32_t>(fm.buf.size());
  for (int i = 0; i < 4; i++) {
    out.push_back((flen >> (8 * i)) & 0xff);
  }
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  return out;
}

// lists of varying lengths 1..8 with empties mixed in (including a trailing empty list), leaf
// values from the varying-delta generator above
inline std::vector<std::vector<int64_t>> delta_test_lists(int n_lists, uint64_t seed = 401)
{
  std::vector<size_t> lengths(n_lists);
  size_t n_leaf = 0;
  for (int i = 0; i < n_lists; i++) {
    bool const empty = (i + 1 == n_lists) || delta_test_rand(seed) % 6 == 0;
    lengths[i]       = empty ? 0 : 1 + delta_test_rand(seed) % 8;
    n_leaf += lengths[i];
  }
  auto const values = delta_test_int64_values(n_leaf, seed);
  std::vector<std::vector<int64_t>> out(n_lists);
  size_t pos = 0;
  for (int i = 0; i < n_lists; i++) {
    out[i].assign(values.begin() + pos, values.begin() + pos + lengths[i]);
    pos += lengths[i];
  }
  return out;
}
