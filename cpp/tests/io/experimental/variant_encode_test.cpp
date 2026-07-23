/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <string>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// Test helpers
// ──────────────────────────────────────────────────────────────────────────────

namespace {

// Byte-level encoding helpers (host-side, matching the GPU implementation)

inline std::vector<uint8_t> enc_null() { return {0x00}; }

inline std::vector<uint8_t> enc_int8(int8_t v) { return {0x0c, static_cast<uint8_t>(v)}; }

inline std::vector<uint8_t> enc_int16(int16_t v)
{
  auto const u = static_cast<uint16_t>(v);
  return {0x10, static_cast<uint8_t>(u & 0xffu), static_cast<uint8_t>((u >> 8) & 0xffu)};
}

inline std::vector<uint8_t> enc_int32(int32_t v)
{
  auto const u = static_cast<uint32_t>(v);
  return {0x14,
          static_cast<uint8_t>(u & 0xffu),
          static_cast<uint8_t>((u >> 8) & 0xffu),
          static_cast<uint8_t>((u >> 16) & 0xffu),
          static_cast<uint8_t>((u >> 24) & 0xffu)};
}

inline std::vector<uint8_t> enc_int64(int64_t v)
{
  auto const u = static_cast<uint64_t>(v);
  std::vector<uint8_t> out{0x18};
  for (int b = 0; b < 8; b++) {
    out.push_back(static_cast<uint8_t>((u >> (8 * b)) & 0xffu));
  }
  return out;
}

inline std::vector<uint8_t> enc_short_string(std::string_view s)
{
  std::vector<uint8_t> out{static_cast<uint8_t>(0x01 | (s.size() << 2))};
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

inline std::vector<uint8_t> enc_long_string(std::string_view s)
{
  auto const len = static_cast<uint32_t>(s.size());
  std::vector<uint8_t> out{0x40};
  for (int b = 0; b < 4; b++) {
    out.push_back(static_cast<uint8_t>((len >> (8 * b)) & 0xffu));
  }
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

inline std::vector<uint8_t> enc_string(std::string_view s)
{
  return s.size() < 64 ? enc_short_string(s) : enc_long_string(s);
}

// Build VARIANT metadata blob for a sorted list of key names (offset_size=1 assumed; keys must
// have total byte length <= 255).
inline std::vector<uint8_t> build_metadata(std::vector<std::string> const& sorted_keys)
{
  std::vector<uint8_t> out{0x01, static_cast<uint8_t>(sorted_keys.size())};
  std::vector<uint8_t> offsets{0x00};
  uint8_t running = 0;
  for (auto const& k : sorted_keys) {
    running = static_cast<uint8_t>(running + k.size());
    offsets.push_back(running);
  }
  out.insert(out.end(), offsets.begin(), offsets.end());
  for (auto const& k : sorted_keys) {
    out.insert(out.end(), k.begin(), k.end());
  }
  return out;
}

// Build an object VARIANT value blob with field_id_size=1, field_offset_size=4.
// `field_values` must be in the same order as sorted_keys (IDs 0..N-1).
inline std::vector<uint8_t> build_object_value(
  std::vector<std::vector<uint8_t>> const& field_values)
{
  auto const N = static_cast<int>(field_values.size());
  std::vector<uint8_t> out;

  // value_metadata = 0x0e: object, field_id_size=1, field_offset_size=4, num_elements_size=1
  out.push_back(0x0e);
  out.push_back(static_cast<uint8_t>(N));

  // field_ids = 0, 1, ..., N-1
  for (int i = 0; i < N; i++) {
    out.push_back(static_cast<uint8_t>(i));
  }

  // field_offsets[0..N] (4 bytes each LE)
  uint32_t running = 0;
  for (int i = 0; i <= N; i++) {
    for (int b = 0; b < 4; b++) {
      out.push_back(static_cast<uint8_t>((running >> (8 * b)) & 0xffu));
    }
    if (i < N) { running += static_cast<uint32_t>(field_values[i].size()); }
  }

  // field values
  for (auto const& fv : field_values) {
    out.insert(out.end(), fv.begin(), fv.end());
  }
  return out;
}

// Build a multi-row VARIANT struct column from explicit per-row (meta, value) byte vectors.
inline cudf::test::structs_column_wrapper make_variant_column(
  std::vector<std::vector<uint8_t>> const& meta_rows,
  std::vector<std::vector<uint8_t>> const& val_rows)
{
  auto build = [](std::vector<std::vector<uint8_t>> const& rows) {
    auto n = static_cast<cudf::size_type>(rows.size());
    std::vector<int32_t> offsets(n + 1, 0);
    std::vector<uint8_t> flat;
    for (int i = 0; i < n; i++) {
      flat.insert(flat.end(), rows[i].begin(), rows[i].end());
      offsets[i + 1] = static_cast<int32_t>(flat.size());
    }
    auto offs_col =
      cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
    auto data_col =
      cudf::test::fixed_width_column_wrapper<uint8_t>(flat.begin(), flat.end()).release();
    return cudf::make_lists_column(n, std::move(offs_col), std::move(data_col), 0, {});
  };

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(build(meta_rows));
  children.push_back(build(val_rows));
  return cudf::test::structs_column_wrapper{std::move(children)};
}

// Return the metadata child column view (child 0) of a VARIANT struct column.
inline cudf::column_view meta_child(cudf::column_view const& variant_col)
{
  return cudf::structs_column_view{variant_col}.child(0);
}

// Return the value child column view (child 1) of a VARIANT struct column.
inline cudf::column_view val_child(cudf::column_view const& variant_col)
{
  return cudf::structs_column_view{variant_col}.child(1);
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────────
// encode_strings_to_variant tests
// ──────────────────────────────────────────────────────────────────────────────

struct EncodeStringsToVariantTest : public cudf::test::BaseFixture {};

TEST_F(EncodeStringsToVariantTest, EmptyInput)
{
  cudf::test::strings_column_wrapper input{};
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());
  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
}

TEST_F(EncodeStringsToVariantTest, WrongTypeThrows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::encode_strings_to_variant(
                 input, cudf::test::get_default_stream())),
               std::invalid_argument);
}

TEST_F(EncodeStringsToVariantTest, ShortString)
{
  cudf::test::strings_column_wrapper input{"hi"};
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 0);

  auto const expected_val  = enc_short_string("hi");
  auto const expected_meta = std::vector<uint8_t>{0x01, 0x00, 0x00};
  auto expected            = make_variant_column({expected_meta}, {expected_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeStringsToVariantTest, LongString)
{
  // A string of exactly 64 bytes should use long_string encoding
  std::string const s(64, 'x');
  cudf::test::strings_column_wrapper input{s};
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  auto const expected_val  = enc_long_string(s);
  auto const expected_meta = std::vector<uint8_t>{0x01, 0x00, 0x00};
  auto expected            = make_variant_column({expected_meta}, {expected_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeStringsToVariantTest, NullInputProducesNullStructRow)
{
  cudf::test::strings_column_wrapper input({"hello", "", "world"}, {true, false, true});
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 3);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(EncodeStringsToVariantTest, AllNullInput)
{
  cudf::test::strings_column_wrapper input({"", "", ""}, {false, false, false});
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  EXPECT_EQ(got->null_count(), 3);
}

TEST_F(EncodeStringsToVariantTest, MultiRowMixedLengths)
{
  std::string const short_s = "abc";                  // len 3 < 64
  std::string const long_s  = std::string(100, 'z');  // len 100 >= 64

  cudf::test::strings_column_wrapper input{short_s, long_s};
  auto got = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 2);
  EXPECT_EQ(got->null_count(), 0);

  // Verify the value child has the right encoding for each row
  auto const ev0 = enc_short_string(short_s);
  auto const ev1 = enc_long_string(long_s);
  auto expected  = make_variant_column(
    {std::vector<uint8_t>{0x01, 0x00, 0x00}, std::vector<uint8_t>{0x01, 0x00, 0x00}}, {ev0, ev1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeStringsToVariantTest, RoundtripWithCastVariant)
{
  // encode then decode: cast_variant should recover the original strings
  cudf::test::strings_column_wrapper input{"foo", "bar", "baz"};
  auto variant = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::cast_variant(
    val_child(*variant), cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, input);
}

TEST_F(EncodeStringsToVariantTest, RoundtripLongStringWithCastVariant)
{
  std::string const long_s = std::string(128, 'a');
  cudf::test::strings_column_wrapper input{long_s};
  auto variant = cudf::io::parquet::experimental::encode_strings_to_variant(
    input, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::cast_variant(
    val_child(*variant), cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, input);
}

// ──────────────────────────────────────────────────────────────────────────────
// encode_variant tests
// ──────────────────────────────────────────────────────────────────────────────

struct EncodeVariantTest : public cudf::test::BaseFixture {};

TEST_F(EncodeVariantTest, EmptyTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"x"};
  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->size(), 0);
}

TEST_F(EncodeVariantTest, UnsupportedTypeThrows)
{
  cudf::test::fixed_width_column_wrapper<float> col{1.0f, 2.0f};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"f"};
  EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::encode_variant(
                 tbl, names, cudf::test::get_default_stream())),
               std::invalid_argument);
}

TEST_F(EncodeVariantTest, ColumnNamesMismatchThrows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{1, 2};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"a", "b"};  // 2 names, 1 column
  EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::encode_variant(
                 tbl, names, cudf::test::get_default_stream())),
               std::invalid_argument);
}

TEST_F(EncodeVariantTest, SingleInt32Column)
{
  // Table: col "x" = [42]
  cudf::test::fixed_width_column_wrapper<int32_t> col{42};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"x"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 0);

  // metadata: {0x01, num_keys=1, offsets=[0,1], "x"} = {0x01, 0x01, 0x00, 0x01, 'x'}
  auto const exp_meta = build_metadata({"x"});
  // value: object with 1 field, field_id=0, field_value=enc_int32(42)
  auto const exp_val = build_object_value({enc_int32(42)});

  auto expected = make_variant_column({exp_meta}, {exp_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, SingleStringColumn)
{
  cudf::test::strings_column_wrapper col{"hello"};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"s"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto const exp_meta = build_metadata({"s"});
  auto const exp_val  = build_object_value({enc_short_string("hello")});

  auto expected = make_variant_column({exp_meta}, {exp_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, NullValueEncodedAsVariantNull)
{
  // A null INT32 value should produce a null primitive (0x00) inside the object
  cudf::test::fixed_width_column_wrapper<int32_t> col({0}, {false});
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"x"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->null_count(), 0);  // struct row is never null

  auto const exp_meta = build_metadata({"x"});
  auto const exp_val  = build_object_value({enc_null()});

  auto expected = make_variant_column({exp_meta}, {exp_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, MultipleColumns_SortedNames)
{
  // Columns "b", "a": the output dictionary must sort to ["a","b"]
  cudf::test::fixed_width_column_wrapper<int32_t> col_b{10};
  cudf::test::strings_column_wrapper col_a{"hi"};
  cudf::table_view tbl{{col_b, col_a}};
  std::vector<std::string> names{"b", "a"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  // Dictionary is sorted: ["a", "b"]. sort_order = [1 (col "a"), 0 (col "b")].
  // Field 0 (id=0, name="a") → col_a value = enc_short_string("hi")
  // Field 1 (id=1, name="b") → col_b value = enc_int32(10)
  auto const exp_meta = build_metadata({"a", "b"});
  auto const exp_val  = build_object_value({enc_short_string("hi"), enc_int32(10)});

  auto expected = make_variant_column({exp_meta}, {exp_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, AllIntegerTypes)
{
  cudf::test::fixed_width_column_wrapper<int8_t> c8{int8_t{7}};
  cudf::test::fixed_width_column_wrapper<int16_t> c16{int16_t{300}};
  cudf::test::fixed_width_column_wrapper<int32_t> c32{int32_t{70000}};
  cudf::test::fixed_width_column_wrapper<int64_t> c64{int64_t{5000000000LL}};
  cudf::table_view tbl{{c8, c16, c32, c64}};
  std::vector<std::string> names{"i8", "i16", "i32", "i64"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 0);

  // sorted names: i16, i32, i64, i8
  auto const exp_meta = build_metadata({"i16", "i32", "i64", "i8"});
  auto const exp_val =
    build_object_value({enc_int16(300), enc_int32(70000), enc_int64(5000000000LL), enc_int8(7)});

  auto expected = make_variant_column({exp_meta}, {exp_val});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, MultiRow)
{
  // 3 rows: col "a" (int32), col "b" (string)
  cudf::test::fixed_width_column_wrapper<int32_t> col_a{1, 2, 3};
  cudf::test::strings_column_wrapper col_b{"x", "yy", "zzz"};
  cudf::table_view tbl{{col_a, col_b}};
  std::vector<std::string> names{"a", "b"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->size(), 3);
  EXPECT_EQ(got->null_count(), 0);

  auto const meta     = build_metadata({"a", "b"});
  auto const exp_val0 = build_object_value({enc_int32(1), enc_short_string("x")});
  auto const exp_val1 = build_object_value({enc_int32(2), enc_short_string("yy")});
  auto const exp_val2 = build_object_value({enc_int32(3), enc_short_string("zzz")});

  auto expected = make_variant_column({meta, meta, meta}, {exp_val0, exp_val1, exp_val2});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, MixedNullsInMultiColumn)
{
  // col "a": [1, null, 3]
  // col "b": ["x", "y", null]
  cudf::test::fixed_width_column_wrapper<int32_t> col_a({1, 0, 3}, {true, false, true});
  cudf::test::strings_column_wrapper col_b({"x", "y", ""}, {true, true, false});
  cudf::table_view tbl{{col_a, col_b}};
  std::vector<std::string> names{"a", "b"};

  auto got =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  EXPECT_EQ(got->null_count(), 0);  // struct rows never null in encode_variant

  auto const meta     = build_metadata({"a", "b"});
  auto const exp_val0 = build_object_value({enc_int32(1), enc_short_string("x")});
  auto const exp_val1 = build_object_value({enc_null(), enc_short_string("y")});
  auto const exp_val2 = build_object_value({enc_int32(3), enc_null()});

  auto expected = make_variant_column({meta, meta, meta}, {exp_val0, exp_val1, exp_val2});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(EncodeVariantTest, RoundtripInt32WithExtractVariant)
{
  // Encode, then extract "x" and decode as INT32
  cudf::test::fixed_width_column_wrapper<int32_t> col{10, 20, 30};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"x"};

  auto variant =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, col);
}

TEST_F(EncodeVariantTest, RoundtripStringWithExtractVariant)
{
  cudf::test::strings_column_wrapper col{"alpha", "beta", "gamma"};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"s"};

  auto variant =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "s", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, col);
}

TEST_F(EncodeVariantTest, RoundtripNullsWithExtractVariant)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({5, 0, 15}, {true, false, true});
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"v"};

  auto variant =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "v", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, col);
}

TEST_F(EncodeVariantTest, RoundtripMultiColumnWithExtractVariant)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col_a{100, 200, 300};
  cudf::test::strings_column_wrapper col_b{"p", "q", "r"};
  cudf::table_view tbl{{col_a, col_b}};
  std::vector<std::string> names{"a", "b"};

  auto variant =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto decoded_a = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "a", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  auto decoded_b = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "b", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded_a, col_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded_b, col_b);
}

TEST_F(EncodeVariantTest, LargeStringFieldRoundtrip)
{
  // Strings longer than 64 bytes must use long_string encoding in the object field
  std::string const long_s(200, 'k');
  cudf::test::strings_column_wrapper col{long_s};
  cudf::table_view tbl{{col}};
  std::vector<std::string> names{"big"};

  auto variant =
    cudf::io::parquet::experimental::encode_variant(tbl, names, cudf::test::get_default_stream());

  auto decoded = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "big", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, col);
}
