/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apache_variant_fixtures.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace avf = cudf::test::apache_variant_fixtures;

namespace {

// Build a struct `column_view` over (metadata, value) without copying.
inline cudf::column_view wrap_variant_view(cudf::column_view const& metadata,
                                           cudf::column_view const& value)
{
  CUDF_EXPECTS(metadata.size() == value.size(),
               "metadata and value columns must have the same number of rows");
  return cudf::column_view{cudf::data_type{cudf::type_id::STRUCT},
                           value.size(),
                           nullptr,
                           nullptr,
                           0,
                           0,
                           {metadata, value}};
}

// Wrap a single-row (metadata, value) pair as a VARIANT struct column.
inline cudf::test::structs_column_wrapper wrap_single_variant(std::vector<uint8_t> const& meta,
                                                              std::vector<uint8_t> const& val)
{
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

// Wrap an Apache parquet-testing fixture into a single-row VARIANT struct column.
template <std::size_t M, std::size_t V>
cudf::test::structs_column_wrapper make_apache_variant(avf::fixture<M, V> const& f)
{
  cudf::test::lists_column_wrapper<uint8_t> m(f.metadata.begin(), f.metadata.end());
  cudf::test::lists_column_wrapper<uint8_t> v(f.value.begin(), f.value.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

// Three-row VARIANT fixture reused by multiple multi-row tests below.
//   Row 0: dict {x,y}, value { x: INT32(7),  y: "hi"  }
//   Row 1: dict {x,z}, value { x: INT32(42), z: INT32(99) }
//   Row 2: dict {y},   value { y: "zzz" }
inline cudf::test::structs_column_wrapper make_xyz_three_row_variant()
{
  std::vector<uint8_t> const m1 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y'};
  std::vector<uint8_t> const v1 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  std::vector<uint8_t> const m2 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'z'};
  // clang-format off
  std::vector<uint8_t> const v2 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x0a,
    0x14, 0x2a, 0x00, 0x00, 0x00,
    0x14, 0x63, 0x00, 0x00, 0x00};
  // clang-format on
  std::vector<uint8_t> const m3 = {0x01, 0x01, 0x00, 0x01, 'y'};
  std::vector<uint8_t> const v3 = {0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z'};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m1.begin(), m1.end()}, {m2.begin(), m2.end()}, {m3.begin(), m3.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v1.begin(), v1.end()}, {v2.begin(), v2.end()}, {v3.begin(), v3.end()}};
  return cudf::test::structs_column_wrapper{{meta, val}};
}

}  // namespace

struct ExtractVariantFieldTest : public cudf::test::BaseFixture {};

TEST_F(ExtractVariantFieldTest, NullStructRow)
{
  std::vector<uint8_t> const m = {0x01, 0x01, 0x00, 0x01, 'x'};
  // Row 0 object also exercises a mismatched field_id_size (2 bytes) vs field_offset_size (1 byte)
  std::vector<uint8_t> const v = {0x12, 0x01, 0x00, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta{{m.begin(), m.end()}, {0x00}};
  cudf::test::lists_column_wrapper<uint8_t> val{{v.begin(), v.end()}, {0x00}};
  // Use the validity vector to mask the second row null.
  cudf::test::structs_column_wrapper col{{meta, val}, std::vector<bool>{true, false}};

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, NonObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Primitive int32 only (not wrapped in object)
  std::vector<uint8_t> const valb = {0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                        = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, InvalidMetadataYieldsNull)
{
  // Too short to be valid VARIANT metadata V1
  std::vector<uint8_t> const metab = {0x02};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                         = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, UnsupportedMetadataVersionYieldsNull)
{
  // Variant version != v1 must produce null on field lookup.
  std::vector<uint8_t> const metab = {0x02, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                         = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, TruncatedObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Object header only (truncated)
  std::vector<uint8_t> const valb = {0x02};
  auto col                        = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, MultiRow)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();
  auto x      = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp({7, 42, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);

  auto y = cudf::io::parquet::experimental::extract_variant_field(
    col, "y", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper y_exp({"hi", "", "zzz"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*y, y_exp);

  auto z = cudf::io::parquet::experimental::extract_variant_field(
    col, "z", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> z_exp({0, 99, 0}, {false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*z, z_exp);
}

TEST_F(ExtractVariantFieldTest, SlicedInput)
{
  // Slice rows [1, 3); extracted column must reflect the slice, not the underlying child rows.
  auto const col    = make_xyz_three_row_variant();
  auto const sliced = cudf::slice(col, {1, 3}).front();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    sliced, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({42, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveStringFields)
{
  auto col     = make_apache_variant(avf::object_primitive);
  auto stream  = cudf::test::get_default_stream();
  auto const s = cudf::data_type{cudf::type_id::STRING};

  for (auto const& [field, expected_str] :
       {std::pair{"string_field", "Apache Parquet"},
        std::pair{"timestamp_field", "2025-04-16T12:34:56.78"}}) {
    SCOPED_TRACE(std::string{"field: "} + field);
    auto got = cudf::io::parquet::experimental::extract_variant_field(col, field, s, stream);
    cudf::test::strings_column_wrapper expected({expected_str});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveNullCases)
{
  auto col     = make_apache_variant(avf::object_primitive);
  auto stream  = cudf::test::get_default_stream();
  auto const s = cudf::data_type{cudf::type_id::STRING};

  for (auto const& field : {"no_such_field", "null_field"}) {
    SCOPED_TRACE(std::string{"field: "} + field);
    auto got = cudf::io::parquet::experimental::extract_variant_field(col, field, s, stream);
    ASSERT_EQ(got->size(), 1);
    EXPECT_EQ(got->null_count(), 1);
  }
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveIntField)
{
  auto col = make_apache_variant(avf::object_primitive);
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "int_field", cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNested)
{
  auto col         = make_apache_variant(avf::object_nested);
  auto stream      = cudf::test::get_default_stream();
  auto const check = [&](char const* path, auto expected_val) {
    using T = decltype(expected_val);
    SCOPED_TRACE(std::string{"path: "} + path);
    if constexpr (std::is_same_v<T, char const*>) {
      auto got = cudf::io::parquet::experimental::extract_variant_field(
        col, path, cudf::data_type{cudf::type_id::STRING}, stream);
      cudf::test::strings_column_wrapper expected({expected_val});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
    } else {
      auto got = cudf::io::parquet::experimental::extract_variant_field(
        col, path, cudf::data_type{cudf::type_to_id<T>()}, stream);
      cudf::test::fixed_width_column_wrapper<T> expected{expected_val};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
    }
  };

  check("$.observation.location", "In the Volcano");
  check("$.observation.time", "12:34:56");
  check("$.observation.value.temperature", int8_t{123});
  check("$.species.name", "lava monster");
  check("$.species.population", int16_t{6789});
  check("$.id", int8_t{1});
}

TEST_F(ExtractVariantFieldTest, ApacheObjectEmpty)
{
  auto col = make_apache_variant(avf::object_empty);
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "foo", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNestedChainedCalls)
{
  auto col    = make_apache_variant(avf::object_nested);
  auto stream = cudf::test::get_default_stream();

  auto single = cudf::io::parquet::experimental::get_variant_field(
    col, "$.observation.value.temperature", stream);

  auto const meta_v = cudf::structs_column_view{col}.get_sliced_child(0, stream);
  auto obs  = cudf::io::parquet::experimental::get_variant_field(col, "observation", stream);
  auto vobj = cudf::io::parquet::experimental::get_variant_field(
    wrap_variant_view(meta_v, obs->view()), "value", stream);
  auto chained = cudf::io::parquet::experimental::get_variant_field(
    wrap_variant_view(meta_v, vobj->view()), "temperature", stream);

  EXPECT_EQ(single->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(chained->type().id(), cudf::type_id::LIST);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*single, *chained);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNestedMissingIntermediate)
{
  auto col    = make_apache_variant(avf::object_nested);
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.species.nope", cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"donotread"}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, NestedPathNonObjectIntermediate)
{
  // Dict = {a, b}; value = { a: INT32(5), b: "hi" }
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'a', 'b'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x05, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};

  auto col = wrap_single_variant(metab, valb);
  // Descending into "a" fails because it is a primitive, not an object.
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.a.b", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, BareNameEqualsDollarPath)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();

  auto bare   = cudf::io::parquet::experimental::get_variant_field(col, "x", stream);
  auto dollar = cudf::io::parquet::experimental::get_variant_field(col, "$.x", stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

namespace {

// INT32 primitive blob: header 0x14, little-endian 4-byte payload.
inline std::vector<uint8_t> enc_int32(int32_t v)
{
  auto const u = static_cast<uint32_t>(v);
  return {0x14,
          static_cast<uint8_t>(u & 0xff),
          static_cast<uint8_t>((u >> 8) & 0xff),
          static_cast<uint8_t>((u >> 16) & 0xff),
          static_cast<uint8_t>((u >> 24) & 0xff)};
}

// Short-string primitive blob (single-byte header).
inline std::vector<uint8_t> enc_short_string(std::string_view s)
{
  CUDF_EXPECTS(s.size() < 64, "short-string length must fit in 6 bits of the single-byte header");
  std::vector<uint8_t> out{static_cast<uint8_t>(0x01 | (s.size() << 2))};
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// Build a single-field object value wrapping `inner` under field id `fid`.
// field_off_size=1, field_id_size=1, is_large=false.
inline std::vector<uint8_t> build_single_field_object(uint8_t fid,
                                                      std::vector<uint8_t> const& inner)
{
  CUDF_EXPECTS(inner.size() < 256, "inner blob too large for 1-byte offset header");
  // Header, num_elements, field_id, offset 0, sentinel = inner.size().
  std::vector<uint8_t> out{0x02, 0x01, fid, 0x00, static_cast<uint8_t>(inner.size())};
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

// Build a VARIANT object blob with `n_fields` fields.  Field ids are 0..n_fields-1
// (in ascending order, matching the dictionary positions) and each field holds a bare INT32 equal
// to its field id.  Uses 1-byte field_id_size and 1-byte field_off_size; n_fields must be
// <= 51 so the total value bytes (5 * n_fields) still fit in 1-byte offsets.
inline std::vector<uint8_t> build_sequential_int32_object(int n_fields)
{
  std::vector<uint8_t> out{0x02, static_cast<uint8_t>(n_fields)};
  for (int fid = 0; fid < n_fields; ++fid) {
    out.push_back(static_cast<uint8_t>(fid));
  }
  for (int i = 0; i <= n_fields; ++i) {
    out.push_back(static_cast<uint8_t>(i * 5));
  }
  for (int fid = 0; fid < n_fields; ++fid) {
    auto const v = enc_int32(fid);
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

// Lexicographically ordered dictionary of N zero-padded two-digit keys "k<NN>".
inline std::vector<std::string> make_numeric_keys(int n)
{
  std::vector<std::string> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::array<char, 8> buf{};
    std::snprintf(buf.data(), buf.size(), "k%02d", i);
    out.emplace_back(buf.data());
  }
  return out;
}

// Wrap per-row (metadata, value) byte vectors into a VARIANT struct column.  Built
// with make_lists_column + structs_column_wrapper directly so the helper stays
// self-contained within this test file for dynamic row counts.
inline cudf::test::structs_column_wrapper wrap_multi_row_variant(
  std::vector<std::vector<uint8_t>> const& meta_rows,
  std::vector<std::vector<uint8_t>> const& val_rows)
{
  auto build_list = [](std::vector<std::vector<uint8_t>> const& rows) {
    auto const n = static_cast<cudf::size_type>(rows.size());
    std::vector<int32_t> offsets(n + 1, 0);
    std::vector<uint8_t> flat;
    for (cudf::size_type i = 0; i < n; ++i) {
      flat.insert(flat.end(), rows[i].begin(), rows[i].end());
      offsets[i + 1] = static_cast<int32_t>(flat.size());
    }
    auto offs =
      cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
    auto data = cudf::test::fixed_width_column_wrapper<uint8_t>(flat.begin(), flat.end()).release();
    return cudf::make_lists_column(n, std::move(offs), std::move(data), 0, {});
  };
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(build_list(meta_rows));
  children.emplace_back(build_list(val_rows));
  return cudf::test::structs_column_wrapper{std::move(children)};
}

// Build a metadata blob (version 1, offset_size=1) for the given ordered string dictionary.
inline std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys)
{
  std::vector<uint8_t> out{0x01, static_cast<uint8_t>(keys.size())};

  std::vector<uint8_t> offs{0x00};
  uint8_t running = 0;
  for (auto const& k : keys) {
    running = static_cast<uint8_t>(running + k.size());
    offs.push_back(running);
  }
  out.insert(out.end(), offs.begin(), offs.end());

  for (auto const& k : keys) {
    out.insert(out.end(), k.begin(), k.end());
  }
  return out;
}

}  // namespace

TEST_F(ExtractVariantFieldTest, NestedPathMultiRowMixedNulls)
{
  // Row 0: { 1st: { foo-bar: INT32(1) } } -> path "$.1st.foo-bar" = 1.  Dictionary strings are
  // stored in non-lexicographic order ({"foo-bar", "1st"})
  auto const m0 = build_metadata({"foo-bar", "1st"});
  auto const v0 = build_single_field_object(
    /*fid=1st*/ 1, build_single_field_object(/*fid=foo-bar*/ 0, enc_int32(1)));
  // Row 1: { 1st: INT32(5) } -> non-object intermediate at "1st" -> null
  auto const m1 = build_metadata({"1st"});
  auto const v1 = build_single_field_object(/*fid=1st*/ 0, enc_int32(5));
  // Row 2: { q: INT32(7) } -> key "1st" missing from dict -> null
  auto const m2 = build_metadata({"q"});
  auto const v2 = build_single_field_object(/*fid=q*/ 0, enc_int32(7));

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m0.begin(), m0.end()}, {m1.begin(), m1.end()}, {m2.begin(), m2.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v0.begin(), v0.end()}, {v1.begin(), v1.end()}, {v2.begin(), v2.end()}};
  cudf::test::structs_column_wrapper col{{meta, val}};

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.1st.foo-bar", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0}, {true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, EmptyPathRejected)
{
  auto col    = wrap_single_variant(build_metadata({}), enc_int32(1));
  auto stream = cudf::test::get_default_stream();
  EXPECT_THROW(
    static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, "", stream)),
    std::invalid_argument);
  EXPECT_THROW(
    static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, "$", stream)),
    std::invalid_argument);
  EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::extract_variant_field(
                 col, "", cudf::data_type{cudf::type_id::INT32}, stream)),
               std::invalid_argument);
}

TEST_F(ExtractVariantFieldTest, SyntaxErrors)
{
  auto col    = wrap_single_variant(build_metadata({}), enc_int32(1));
  auto stream = cudf::test::get_default_stream();
  // Only object-key descent is supported — array indexing, bracket steps, and quoted keys should
  // throw, alongside malformed paths.
  for (auto const* bad : {"$..a", "$.a[0]", "$.a[", "$.a[]", "$.", "$['x']", "$.a[*]"}) {
    EXPECT_THROW(
      static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, bad, stream)),
      std::invalid_argument)
      << "path that should have thrown: " << bad;
  }
}

TEST_F(ExtractVariantFieldTest, LargeDictionaryAndObjectScan)
{
  auto const keys = make_numeric_keys(50);
  auto const meta = build_metadata(keys);
  auto const val  = build_sequential_int32_object(50);
  auto col        = wrap_single_variant(meta, val);
  auto stream     = cudf::test::get_default_stream();
  auto const i32  = cudf::data_type{cudf::type_id::INT32};

  // First, middle, and last keys each decode to their own field id.
  auto first = cudf::io::parquet::experimental::extract_variant_field(col, "k00", i32, stream);
  auto mid   = cudf::io::parquet::experimental::extract_variant_field(col, "k24", i32, stream);
  auto last  = cudf::io::parquet::experimental::extract_variant_field(col, "k49", i32, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*first, cudf::test::fixed_width_column_wrapper<int32_t>{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*mid, cudf::test::fixed_width_column_wrapper<int32_t>{24});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*last, cudf::test::fixed_width_column_wrapper<int32_t>{49});
}

TEST_F(ExtractVariantFieldTest, NullsAtDifferentDepths)
{
  std::vector<std::string> const dict = {"a", "b", "c", "d"};  // fids: a=0,b=1,c=2,d=3
  auto const meta                     = build_metadata(dict);

  // Shape 0: intact — {a:{b:{c:{d:"leaf"}}}}
  auto const s0_d    = enc_short_string("leaf");
  auto const s0_cd   = build_single_field_object(/*fid=d*/ 3, s0_d);
  auto const s0_bc   = build_single_field_object(/*fid=c*/ 2, s0_cd);
  auto const s0_ab   = build_single_field_object(/*fid=b*/ 1, s0_bc);
  auto const s0_root = build_single_field_object(/*fid=a*/ 0, s0_ab);

  // Shape 1: missing key at depth 1 — root has "b" but no "a".
  auto const s1_b    = enc_int32(0);
  auto const s1_root = build_single_field_object(/*fid=b*/ 1, s1_b);

  // Shape 2: kind mismatch at depth 3 — {a:{b:INT32(7)}} so descending into "c" fails.
  auto const s2_bval = enc_int32(7);
  auto const s2_ab   = build_single_field_object(/*fid=b*/ 1, s2_bval);
  auto const s2_root = build_single_field_object(/*fid=a*/ 0, s2_ab);

  // Shape 3: missing key at depth 4 — {a:{b:{c:INT32(9)}}} so the final ".d" misses.
  auto const s3_cval = enc_int32(9);
  auto const s3_bc   = build_single_field_object(/*fid=c*/ 2, s3_cval);
  auto const s3_ab   = build_single_field_object(/*fid=b*/ 1, s3_bc);
  auto const s3_root = build_single_field_object(/*fid=a*/ 0, s3_ab);

  std::vector<std::vector<uint8_t> const*> const shapes{&s0_root, &s1_root, &s2_root, &s3_root};

  constexpr int num_rows = 128;
  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  std::vector<char const*> exp_strs(num_rows);
  std::vector<bool> exp_valid(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    int const shape = i % 4;
    val_rows[i]     = *shapes[shape];
    exp_strs[i]     = (shape == 0) ? "leaf" : "";
    exp_valid[i]    = (shape == 0);
  }

  auto col = wrap_multi_row_variant(meta_rows, val_rows);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.a.b.c.d", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected(exp_strs.begin(), exp_strs.end(), exp_valid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, expected);
}

TEST_F(ExtractVariantFieldTest, EmptyInput)
{
  auto const stream  = cudf::test::get_default_stream();
  auto const variant = cudf::empty_like(make_xyz_three_row_variant());

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  EXPECT_EQ(got->type().id(), cudf::type_id::INT32);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
}

struct GetVariantFieldTest : public cudf::test::BaseFixture {};

TEST_F(GetVariantFieldTest, ApacheObjectPrimitive)
{
  auto col    = make_apache_variant(avf::object_primitive);
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::get_variant_field(col, "int_field", stream);

  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(cudf::lists_column_view{got->view()}.child().type().id(), cudf::type_id::UINT8);

  auto casted = cudf::io::parquet::experimental::cast_variant(
    got->view(), cudf::data_type{cudf::type_id::INT8}, stream);
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*casted, expected);
}

TEST_F(GetVariantFieldTest, ApacheObjectPrimitiveMissingKeyAllNull)
{
  auto col = make_apache_variant(avf::object_primitive);
  auto got = cudf::io::parquet::experimental::get_variant_field(
    col, "no_such_field", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(GetVariantFieldTest, GetAndCastMatchesExtract)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();

  auto extract_x = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, stream);

  auto intermediate = cudf::io::parquet::experimental::get_variant_field(col, "x", stream);
  auto two_step_x   = cudf::io::parquet::experimental::cast_variant(
    intermediate->view(), cudf::data_type{cudf::type_id::INT32}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_x, *two_step_x);
}

TEST_F(GetVariantFieldTest, EmptyInput)
{
  auto const stream  = cudf::test::get_default_stream();
  auto const variant = cudf::empty_like(make_xyz_three_row_variant());

  auto got = cudf::io::parquet::experimental::get_variant_field(*variant, "x", stream);
  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
  EXPECT_EQ(cudf::lists_column_view{got->view()}.child().type().id(), cudf::type_id::UINT8);
}

struct CastVariantTest : public cudf::test::BaseFixture {
  template <typename T, std::size_t M, std::size_t V>
  std::unique_ptr<cudf::column> cast_apache_primitive(avf::fixture<M, V> const& fixture,
                                                      rmm::cuda_stream_view stream)
  {
    auto col         = make_apache_variant(fixture);
    auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
    return cudf::io::parquet::experimental::cast_variant(
      value, cudf::data_type{cudf::type_to_id<T>()}, stream);
  }
};

TEST_F(CastVariantTest, ApachePrimitiveInts)
{
  auto stream = cudf::test::get_default_stream();
  {
    auto got = cast_apache_primitive<int8_t>(avf::primitive_int8, stream);
    cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{42}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int16_t>(avf::primitive_int16, stream);
    cudf::test::fixed_width_column_wrapper<int16_t> expected{int16_t{1234}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int32_t>(avf::primitive_int32, stream);
    cudf::test::fixed_width_column_wrapper<int32_t> expected{int32_t{123456}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int64_t>(avf::primitive_int64, stream);
    cudf::test::fixed_width_column_wrapper<int64_t> expected{int64_t{1234567890123456789LL}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(CastVariantTest, ApachePrimitiveFloats)
{
  auto stream = cudf::test::get_default_stream();
  {
    auto got = cast_apache_primitive<float>(avf::primitive_float, stream);
    cudf::test::fixed_width_column_wrapper<float> expected{float{1234567936.0f}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<double>(avf::primitive_double, stream);
    cudf::test::fixed_width_column_wrapper<double> expected{double{1234567890.1234}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(CastVariantTest, ApacheShortString)
{
  auto col         = make_apache_variant(avf::short_string);
  auto stream      = cudf::test::get_default_stream();
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);

  auto got = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::STRING}, stream);

  // Decoded from short_string.value: skip the 1-byte header, take the rest.
  std::string const expected_str(reinterpret_cast<char const*>(avf::short_string.value.data() + 1),
                                 avf::short_string.value.size() - 1);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, ApachePrimitiveString)
{
  auto col         = make_apache_variant(avf::primitive_string);
  auto stream      = cudf::test::get_default_stream();
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);

  auto got = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::STRING}, stream);

  // Long-string layout: 1 header byte + 4-byte LE length + payload.
  std::string const expected_str(
    reinterpret_cast<char const*>(avf::primitive_string.value.data() + 5),
    avf::primitive_string.value.size() - 5);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, MismatchedTypeYieldsNull)
{
  // Casting an object value to a primitive must produce null, not throw
  auto stream      = cudf::test::get_default_stream();
  auto col         = make_apache_variant(avf::object_primitive);
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
  auto got         = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::INT32}, stream);
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(CastVariantTest, EmptyInput)
{
  auto const stream = cudf::test::get_default_stream();
  auto const values =
    cudf::empty_like(cudf::structs_column_view{make_xyz_three_row_variant()}.child(1));

  for (auto const id : {cudf::type_id::INT32,
                        cudf::type_id::STRING,
                        cudf::type_id::FLOAT32,
                        cudf::type_id::FLOAT64}) {
    auto got = cudf::io::parquet::experimental::cast_variant(*values, cudf::data_type{id}, stream);
    EXPECT_EQ(got->type().id(), id);
    EXPECT_EQ(got->size(), 0);
    EXPECT_EQ(got->null_count(), 0);
  }
}
