/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/io/variant.hpp>
#include <cudf/utilities/span.hpp>

#include <string>
#include <vector>

struct VariantExtractTest : public cudf::test::BaseFixture {};

TEST_F(VariantExtractTest, ExtractInt32TopLevelField)
{
  // Metadata: version 1, 1-byte offsets, dictionary {"x"}
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Value: object { "x": 7 } as Variant binary (see parquet-format VariantEncoding.md)
  std::vector<uint8_t> const valb = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};

  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ExtractShortStringField)
{
  // Dictionary {"k"}
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('k')};
  // Object { "k": "hi" } — short string basic_type=1, length 2 in header
  std::vector<uint8_t> const valb = {0x02, 0x01, 0x00, 0x00, 0x03, 0x09, 'h', 'i'};

  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected{"hi"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, NullStructRow)
{
  cudf::test::lists_column_wrapper<uint8_t> meta{
    {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')},
    {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00},
    {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00}};
  cudf::test::structs_column_wrapper struc{{meta, val}, std::vector<bool>{true, false}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, MissingKeyYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "missing", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, WrongDesiredTypeYieldsNull)
{
  // Object holds INT32 at "x"; request STRING
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"donotread"}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, NonObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Primitive int32 only (not wrapped in object)
  std::vector<uint8_t> const valb = {0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, InvalidMetadataYieldsNull)
{
  // Too short to be valid VARIANT metadata v1
  std::vector<uint8_t> const metab = {0x02};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, TruncatedObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Object header only (truncated)
  std::vector<uint8_t> const valb = {0x02};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ObjectTwoFieldsSingleRow)
{
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};
  auto stream = cudf::test::get_default_stream();
  auto x      = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);
  auto k = cudf::io::parquet::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper k_exp{"hi"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*k, k_exp);
}

TEST_F(VariantExtractTest, MultiRowDistinctKeysHandBuilt)
{
  // Same byte payloads as cpp/tests/scripts/parquet_variant_fixture_gen.py (varying dictionaries).
  std::vector<uint8_t> const m1 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const v1 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  std::vector<uint8_t> const m2 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y'};
  std::vector<uint8_t> const v2 = {0x02,
                                   0x02,
                                   0x00,
                                   0x01,
                                   0x00,
                                   0x05,
                                   0x0a,
                                   0x14,
                                   0x2a,
                                   0x00,
                                   0x00,
                                   0x00,
                                   0x14,
                                   0x63,
                                   0x00,
                                   0x00,
                                   0x00};
  std::vector<uint8_t> const m3 = {0x01, 0x01, 0x00, 0x01, 'k'};
  std::vector<uint8_t> const v3 = {0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z'};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m1.begin(), m1.end()}, {m2.begin(), m2.end()}, {m3.begin(), m3.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v1.begin(), v1.end()}, {v2.begin(), v2.end()}, {v3.begin(), v3.end()}};
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto stream = cudf::test::get_default_stream();
  auto x      = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp({7, 42, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);

  auto k = cudf::io::parquet::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper k_exp({"hi", "", "zzz"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*k, k_exp);

  auto y = cudf::io::parquet::extract_variant_field(
    struc, "y", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> y_exp({0, 99, 0}, {false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*y, y_exp);
}

TEST_F(VariantExtractTest, MultiRowMixedKeys)
{
  // Row0: x=1, Row1: object missing x
  std::vector<uint8_t> const m0 = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const v0 = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x01, 0x00, 0x00, 0x00};
  // Row1: only "y" in dictionary and object
  std::vector<uint8_t> const m1 = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('y')};
  std::vector<uint8_t> const v1 = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x02, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta{{m0.begin(), m0.end()}, {m1.begin(), m1.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{{v0.begin(), v0.end()}, {v1.begin(), v1.end()}};
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ExtraShreddingSiblingIgnored)
{
  // Same VARIANT bytes as ObjectTwoFieldsSingleRow; third struct child simulates an extra shredded
  // branch (ignored — extraction uses list<uint8> children 0 and 1 only).
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::fixed_width_column_wrapper<int32_t> shredding_placeholder{0};
  cudf::test::structs_column_wrapper struc{{meta, val, shredding_placeholder}};
  auto stream = cudf::test::get_default_stream();
  auto x      = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);
  auto k = cudf::io::parquet::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper k_exp{"hi"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*k, k_exp);
}

// ---------------------------------------------------------------------------
// Tests for the split API: get_variant_field and cast_variant
// ---------------------------------------------------------------------------

TEST_F(VariantExtractTest, GetVariantFieldReturnsVariantStruct)
{
  // Object { "x": 7, "k": "hi" }
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::get_variant_field(struc, "x", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->num_children(), 2);
  EXPECT_EQ(got->size(), 1);

  auto const child0 = got->view().child(0);
  auto const child1 = got->view().child(1);
  EXPECT_EQ(child0.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(child1.type().id(), cudf::type_id::LIST);

  // The extracted value for "x" should be the INT32 encoding: {0x14, 0x07, 0x00, 0x00, 0x00}
  // Verify it can be decoded via cast_variant
  auto casted = cudf::io::parquet::cast_variant(
    got->view(), cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int32_t> expected{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*casted, expected);
}

TEST_F(VariantExtractTest, GetVariantFieldMissingKeyAllNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got =
    cudf::io::parquet::get_variant_field(struc, "missing", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->null_count(), 1);

  // Metadata child should still be valid
  auto const meta_child = got->view().child(0);
  EXPECT_EQ(meta_child.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(meta_child.size(), 1);
}

TEST_F(VariantExtractTest, CastVariantInt32)
{
  // Build a VARIANT struct where value is a bare INT32 encoding (0x14, 42, 0, 0, 0)
  std::vector<uint8_t> const metab = {0x01, 0x00, 0x00};
  std::vector<uint8_t> const valb  = {0x14, 0x2a, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, CastVariantString)
{
  // Short string "hi" (basic_type=1, header6=2): 0x09, 'h', 'i'
  std::vector<uint8_t> const metab     = {0x01, 0x00, 0x00};
  std::vector<uint8_t> const short_str = {0x09, 'h', 'i'};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(short_str.begin(), short_str.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected{"hi"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, CastVariantLongString)
{
  // Long string (basic_type=0, header6=16): header=0x40, then 4-byte LE length, then chars
  std::string const payload = "hello world!";
  std::vector<uint8_t> valb;
  valb.push_back(0x40);  // (16 << 2) | 0
  auto const slen = static_cast<uint32_t>(payload.size());
  valb.push_back(slen & 0xFF);
  valb.push_back((slen >> 8) & 0xFF);
  valb.push_back((slen >> 16) & 0xFF);
  valb.push_back((slen >> 24) & 0xFF);
  for (auto c : payload) {
    valb.push_back(static_cast<uint8_t>(c));
  }

  std::vector<uint8_t> const metab = {0x01, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected{"hello world!"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, GetThenCastMatchesExtract)
{
  // Multi-row test: verify get_variant_field + cast_variant == extract_variant_field
  std::vector<uint8_t> const m1 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const v1 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  std::vector<uint8_t> const m2 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y'};
  std::vector<uint8_t> const v2 = {0x02,
                                   0x02,
                                   0x00,
                                   0x01,
                                   0x00,
                                   0x05,
                                   0x0a,
                                   0x14,
                                   0x2a,
                                   0x00,
                                   0x00,
                                   0x00,
                                   0x14,
                                   0x63,
                                   0x00,
                                   0x00,
                                   0x00};
  std::vector<uint8_t> const m3 = {0x01, 0x01, 0x00, 0x01, 'k'};
  std::vector<uint8_t> const v3 = {0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z'};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m1.begin(), m1.end()}, {m2.begin(), m2.end()}, {m3.begin(), m3.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v1.begin(), v1.end()}, {v2.begin(), v2.end()}, {v3.begin(), v3.end()}};
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto stream = cudf::test::get_default_stream();

  // extract_variant_field (convenience)
  auto extract_x = cudf::io::parquet::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);

  // get_variant_field + cast_variant (two-step)
  auto intermediate = cudf::io::parquet::get_variant_field(struc, "x", stream);
  auto two_step_x   = cudf::io::parquet::cast_variant(
    intermediate->view(), cudf::data_type{cudf::type_id::INT32}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_x, *two_step_x);

  // Same for string field "k"
  auto extract_k = cudf::io::parquet::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  auto intermediate_k = cudf::io::parquet::get_variant_field(struc, "k", stream);
  auto two_step_k     = cudf::io::parquet::cast_variant(
    intermediate_k->view(), cudf::data_type{cudf::type_id::STRING}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_k, *two_step_k);
}

// ---------------------------------------------------------------------------
// Validation against Apache parquet-testing reference binaries
// https://github.com/apache/parquet-testing/tree/master/variant
// ---------------------------------------------------------------------------

TEST_F(VariantExtractTest, ApachePrimitiveInt32)
{
  // primitive_int32: metadata = empty dict, value = INT32(123456)
  std::vector<uint8_t> const meta = {0x01, 0x00, 0x00};
  std::vector<uint8_t> const val  = {0x14, 0x40, 0xe2, 0x01, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  cudf::test::structs_column_wrapper struc{{m, v}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected{123456};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheShortString)
{
  // short_string: basic_type=1, header6=37, "Less than 64 bytes (❤️ with utf8)"
  // clang-format off
  std::vector<uint8_t> const meta = {0x01, 0x00, 0x00};
  std::vector<uint8_t> const val  = {
    0x95, 0x4c, 0x65, 0x73, 0x73, 0x20, 0x74, 0x68, 0x61, 0x6e, 0x20, 0x36, 0x34,
    0x20, 0x62, 0x79, 0x74, 0x65, 0x73, 0x20, 0x28, 0xe2, 0x9d, 0xa4, 0xef, 0xb8,
    0x8f, 0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x75, 0x74, 0x66, 0x38, 0x29};
  // clang-format on
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  cudf::test::structs_column_wrapper struc{{m, v}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected(
    {"\x4c\x65\x73\x73\x20\x74\x68\x61\x6e\x20\x36\x34"
     "\x20\x62\x79\x74\x65\x73\x20\x28\xe2\x9d\xa4\xef"
     "\xb8\x8f\x20\x77\x69\x74\x68\x20\x75\x74\x66\x38"
     "\x29"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheLongString)
{
  // primitive_string: basic_type=0, header6=16 (long string), 174 UTF-8 bytes with emoji
  // clang-format off
  std::vector<uint8_t> const meta = {0x01, 0x00, 0x00};
  std::vector<uint8_t> const val  = {
    0x40, 0xae, 0x00, 0x00, 0x00, 0x54, 0x68, 0x69, 0x73, 0x20, 0x73, 0x74, 0x72,
    0x69, 0x6e, 0x67, 0x20, 0x69, 0x73, 0x20, 0x6c, 0x6f, 0x6e, 0x67, 0x65, 0x72,
    0x20, 0x74, 0x68, 0x61, 0x6e, 0x20, 0x36, 0x34, 0x20, 0x62, 0x79, 0x74, 0x65,
    0x73, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x74, 0x68, 0x65, 0x72, 0x65, 0x66, 0x6f,
    0x72, 0x65, 0x20, 0x64, 0x6f, 0x65, 0x73, 0x20, 0x6e, 0x6f, 0x74, 0x20, 0x66,
    0x69, 0x74, 0x20, 0x69, 0x6e, 0x20, 0x61, 0x20, 0x73, 0x68, 0x6f, 0x72, 0x74,
    0x5f, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x69,
    0x74, 0x20, 0x61, 0x6c, 0x73, 0x6f, 0x20, 0x69, 0x6e, 0x63, 0x6c, 0x75, 0x64,
    0x65, 0x73, 0x20, 0x73, 0x65, 0x76, 0x65, 0x72, 0x61, 0x6c, 0x20, 0x6e, 0x6f,
    0x6e, 0x20, 0x61, 0x73, 0x63, 0x69, 0x69, 0x20, 0x63, 0x68, 0x61, 0x72, 0x61,
    0x63, 0x74, 0x65, 0x72, 0x73, 0x20, 0x73, 0x75, 0x63, 0x68, 0x20, 0x61, 0x73,
    0x20, 0xf0, 0x9f, 0x90, 0xa2, 0x2c, 0x20, 0xf0, 0x9f, 0x92, 0x96, 0x2c, 0x20,
    0xe2, 0x99, 0xa5, 0xef, 0xb8, 0x8f, 0x2c, 0x20, 0xf0, 0x9f, 0x8e, 0xa3, 0x20,
    0x61, 0x6e, 0x64, 0x20, 0xf0, 0x9f, 0xa4, 0xa6, 0x21, 0x21};
  // clang-format on
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  cudf::test::structs_column_wrapper struc{{m, v}};

  auto got = cudf::io::parquet::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  // The long string decoded from the reference file
  std::string const expected_str(reinterpret_cast<char const*>(val.data() + 5), 174);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitiveExtractString)
{
  // object_primitive from parquet-testing: 7 fields including "string_field" = "Apache Parquet"
  // clang-format off
  std::vector<uint8_t> const meta = {
    0x01, 0x07, 0x00, 0x09, 0x15, 0x27, 0x3a, 0x46, 0x50, 0x5f,
    0x69, 0x6e, 0x74, 0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64,
    0x64, 0x6f, 0x75, 0x62, 0x6c, 0x65, 0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64,
    0x62, 0x6f, 0x6f, 0x6c, 0x65, 0x61, 0x6e, 0x5f, 0x74, 0x72, 0x75, 0x65, 0x5f,
    0x66, 0x69, 0x65, 0x6c, 0x64,
    0x62, 0x6f, 0x6f, 0x6c, 0x65, 0x61, 0x6e, 0x5f, 0x66, 0x61, 0x6c, 0x73, 0x65,
    0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64,
    0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64,
    0x6e, 0x75, 0x6c, 0x6c, 0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64,
    0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x5f, 0x66, 0x69, 0x65,
    0x6c, 0x64};
  std::vector<uint8_t> const val = {
    0x02, 0x07, 0x03, 0x02, 0x01, 0x00, 0x05, 0x04, 0x06,
    0x09, 0x08, 0x02, 0x00, 0x19, 0x0a, 0x1a, 0x31,
    0x0c, 0x01, 0x20, 0x08, 0x15, 0xcd, 0x5b, 0x07, 0x04, 0x08,
    0x39, 0x41, 0x70, 0x61, 0x63, 0x68, 0x65, 0x20, 0x50, 0x61, 0x72, 0x71, 0x75,
    0x65, 0x74, 0x00,
    0x59, 0x32, 0x30, 0x32, 0x35, 0x2d, 0x30, 0x34, 0x2d, 0x31, 0x36, 0x54, 0x31,
    0x32, 0x3a, 0x33, 0x34, 0x3a, 0x35, 0x36, 0x2e, 0x37, 0x38};
  // clang-format on
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  cudf::test::structs_column_wrapper struc{{m, v}};

  auto got = cudf::io::parquet::extract_variant_field(struc,
                                                      "string_field",
                                                      cudf::data_type{cudf::type_id::STRING},
                                                      cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"Apache Parquet"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheNestedGetVariantField)
{
  // object_nested from parquet-testing: {"id":1, "species":{"name":"lava monster",...}, ...}
  // Metadata dictionary: [0]id [1]species [2]name [3]population [4]observation
  //                       [5]time [6]location [7]value [8]temperature [9]humidity
  // Shared across all nesting levels -- this validates nested extraction.
  // clang-format off
  std::vector<uint8_t> const meta = {
    0x01, 0x0a, 0x00, 0x02, 0x09, 0x0d, 0x17, 0x22, 0x26, 0x2e, 0x33, 0x3e, 0x46,
    0x69, 0x64, 0x73, 0x70, 0x65, 0x63, 0x69, 0x65, 0x73, 0x6e, 0x61, 0x6d, 0x65,
    0x70, 0x6f, 0x70, 0x75, 0x6c, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x6f, 0x62, 0x73,
    0x65, 0x72, 0x76, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x6c,
    0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x74,
    0x65, 0x6d, 0x70, 0x65, 0x72, 0x61, 0x74, 0x75, 0x72, 0x65, 0x68, 0x75, 0x6d,
    0x69, 0x64, 0x69, 0x74, 0x79};
  std::vector<uint8_t> const val = {
    0x02, 0x03, 0x00, 0x04, 0x01, 0x00, 0x19, 0x02, 0x46, 0x0c, 0x01, 0x02, 0x02,
    0x02, 0x03, 0x00, 0x0d, 0x10, 0x31, 0x6c, 0x61, 0x76, 0x61, 0x20, 0x6d, 0x6f,
    0x6e, 0x73, 0x74, 0x65, 0x72, 0x10, 0x85, 0x1a, 0x02, 0x03, 0x06, 0x05, 0x07,
    0x09, 0x00, 0x18, 0x24, 0x21, 0x31, 0x32, 0x3a, 0x33, 0x34, 0x3a, 0x35, 0x36,
    0x39, 0x49, 0x6e, 0x20, 0x74, 0x68, 0x65, 0x20, 0x56, 0x6f, 0x6c, 0x63, 0x61,
    0x6e, 0x6f, 0x02, 0x02, 0x09, 0x08, 0x02, 0x00, 0x05, 0x0c, 0x7b, 0x10, 0xc8,
    0x01};
  // clang-format on
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  cudf::test::structs_column_wrapper struc{{m, v}};

  auto stream = cudf::test::get_default_stream();

  // Extract top-level "species" as raw VARIANT, then extract "name" from it, then cast to STRING.
  // This validates: (1) non-monotonic field offsets, (2) shared metadata across nesting levels.
  auto species = cudf::io::parquet::get_variant_field(struc, "species", stream);
  EXPECT_EQ(species->type().id(), cudf::type_id::STRUCT);

  auto name = cudf::io::parquet::get_variant_field(species->view(), "name", stream);
  EXPECT_EQ(name->type().id(), cudf::type_id::STRUCT);

  auto name_str =
    cudf::io::parquet::cast_variant(name->view(), cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"lava monster"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*name_str, expected);
}

namespace {

// --- small Variant blob builders used by the array-indexing / parser tests ---

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

// Short-string primitive blob. Supports length < 64 (single-byte header).
inline std::vector<uint8_t> enc_short_string(std::string_view s)
{
  std::vector<uint8_t> out{static_cast<uint8_t>(0x01 | (s.size() << 2))};
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// Build a Variant array value from concatenated element blobs. Uses off_size=1, is_large=false.
// Total element bytes must be < 256.
inline std::vector<uint8_t> build_array_value(std::vector<std::vector<uint8_t>> const& elements)
{
  // Header + num_elements.
  std::vector<uint8_t> out{0x03, static_cast<uint8_t>(elements.size())};

  // Offsets table: (num_elements + 1) single-byte entries starting at 0.
  std::vector<uint8_t> offs{0x00};
  uint8_t running = 0;
  for (auto const& e : elements) {
    running = static_cast<uint8_t>(running + e.size());
    offs.push_back(running);
  }
  out.insert(out.end(), offs.begin(), offs.end());

  for (auto const& e : elements) {
    out.insert(out.end(), e.begin(), e.end());
  }
  return out;
}

// Build a single-field object value wrapping `inner` under field id `fid`.
// field_off_size=1, field_id_size=1, is_large=false; inner.size() must be < 256.
inline std::vector<uint8_t> build_single_field_object(uint8_t fid,
                                                      std::vector<uint8_t> const& inner)
{
  // Header, num_elements, field_id, offset 0, sentinel = inner.size().
  std::vector<uint8_t> out{0x02, 0x01, fid, 0x00, static_cast<uint8_t>(inner.size())};
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

// Build a metadata blob (version 1, offset_size=1) for the given ordered string dictionary.
// Caller is responsible for sorting keys lexicographically if strict Variant compliance is wanted.
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

// Wrap a single-row (metadata, value) pair as a VARIANT struct column.
inline cudf::test::structs_column_wrapper wrap_single_variant(std::vector<uint8_t> const& meta,
                                                              std::vector<uint8_t> const& val)
{
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

// Shared fixture: Apache parquet-testing object_nested blob. Dictionary:
//   [0]id [1]species [2]name [3]population [4]observation
//   [5]time [6]location [7]value [8]temperature [9]humidity
// Value: { id:1, species:{name:"lava monster", population:6789},
//          observation:{time:"12:34:56", location:"In the Volcano",
//                       value:{temperature:123, humidity:456}} }
struct apache_nested_fixture {
  std::vector<uint8_t> meta;
  std::vector<uint8_t> val;
};

apache_nested_fixture make_apache_nested_fixture()
{
  // clang-format off
  std::vector<uint8_t> const meta = {
    0x01, 0x0a, 0x00, 0x02, 0x09, 0x0d, 0x17, 0x22, 0x26, 0x2e, 0x33, 0x3e, 0x46,
    0x69, 0x64, 0x73, 0x70, 0x65, 0x63, 0x69, 0x65, 0x73, 0x6e, 0x61, 0x6d, 0x65,
    0x70, 0x6f, 0x70, 0x75, 0x6c, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x6f, 0x62, 0x73,
    0x65, 0x72, 0x76, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x6c,
    0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x74,
    0x65, 0x6d, 0x70, 0x65, 0x72, 0x61, 0x74, 0x75, 0x72, 0x65, 0x68, 0x75, 0x6d,
    0x69, 0x64, 0x69, 0x74, 0x79};
  std::vector<uint8_t> const val = {
    0x02, 0x03, 0x00, 0x04, 0x01, 0x00, 0x19, 0x02, 0x46, 0x0c, 0x01, 0x02, 0x02,
    0x02, 0x03, 0x00, 0x0d, 0x10, 0x31, 0x6c, 0x61, 0x76, 0x61, 0x20, 0x6d, 0x6f,
    0x6e, 0x73, 0x74, 0x65, 0x72, 0x10, 0x85, 0x1a, 0x02, 0x03, 0x06, 0x05, 0x07,
    0x09, 0x00, 0x18, 0x24, 0x21, 0x31, 0x32, 0x3a, 0x33, 0x34, 0x3a, 0x35, 0x36,
    0x39, 0x49, 0x6e, 0x20, 0x74, 0x68, 0x65, 0x20, 0x56, 0x6f, 0x6c, 0x63, 0x61,
    0x6e, 0x6f, 0x02, 0x02, 0x09, 0x08, 0x02, 0x00, 0x05, 0x0c, 0x7b, 0x10, 0xc8,
    0x01};
  // clang-format on
  return {meta, val};
}

cudf::test::structs_column_wrapper make_apache_nested_col()
{
  auto const f = make_apache_nested_fixture();
  cudf::test::lists_column_wrapper<uint8_t> m(f.meta.begin(), f.meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(f.val.begin(), f.val.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

}  // namespace

TEST_F(VariantExtractTest, GetNestedPathSpeciesName)
{
  auto col    = make_apache_nested_col();
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::extract_variant_field(
    col, "$.species.name", cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"lava monster"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);

  // Parity: single-call path equals chained calls.
  auto chained_species = cudf::io::parquet::get_variant_field(col, "species", stream);
  auto chained_name = cudf::io::parquet::get_variant_field(chained_species->view(), "name", stream);
  auto chained_str  = cudf::io::parquet::cast_variant(
    chained_name->view(), cudf::data_type{cudf::type_id::STRING}, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, *chained_str);
}

TEST_F(VariantExtractTest, GetNestedPathThreeLevel)
{
  auto col    = make_apache_nested_col();
  auto stream = cudf::test::get_default_stream();

  // Depth 3: observation.value.temperature. The encoded value is INT8 (header 0x0c, byte 0x7b),
  // which cast_variant does not support, so we compare raw VARIANT bytes against the chained
  // result.
  auto single =
    cudf::io::parquet::get_variant_field(col, "$.observation.value.temperature", stream);

  auto obs     = cudf::io::parquet::get_variant_field(col, "observation", stream);
  auto vobj    = cudf::io::parquet::get_variant_field(obs->view(), "value", stream);
  auto chained = cudf::io::parquet::get_variant_field(vobj->view(), "temperature", stream);

  // Both columns must be VARIANT structs with the same value child contents.
  EXPECT_EQ(single->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(chained->type().id(), cudf::type_id::STRUCT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(single->view().child(1), chained->view().child(1));
}

TEST_F(VariantExtractTest, GetNestedPathMissingIntermediate)
{
  auto col    = make_apache_nested_col();
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::extract_variant_field(
    col, "$.species.nope", cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"donotread"}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, GetNestedPathNonObjectIntermediate)
{
  // Dict = {a, b}; value = { a: INT32(5), b: "hi" }. Descending into "a" fails because it is
  // a primitive, not an object.
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'a', 'b'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x05, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};

  cudf::test::lists_column_wrapper<uint8_t> meta(metab.begin(), metab.end());
  cudf::test::lists_column_wrapper<uint8_t> val(valb.begin(), valb.end());
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "$.a.b", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, GetNestedPathEmptyThrows)
{
  auto col = make_apache_nested_col();
  EXPECT_THROW(cudf::io::parquet::get_variant_field(col, "", cudf::test::get_default_stream()),
               std::invalid_argument);
  EXPECT_THROW(cudf::io::parquet::extract_variant_field(
                 col, "", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream()),
               std::invalid_argument);
}

TEST_F(VariantExtractTest, GetNestedPathSingleKeyMatchesOldApi)
{
  // Multi-row fixture reused from MultiRowDistinctKeysHandBuilt.
  std::vector<uint8_t> const m1 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'k'};
  std::vector<uint8_t> const v1 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  std::vector<uint8_t> const m2 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y'};
  std::vector<uint8_t> const v2 = {0x02,
                                   0x02,
                                   0x00,
                                   0x01,
                                   0x00,
                                   0x05,
                                   0x0a,
                                   0x14,
                                   0x2a,
                                   0x00,
                                   0x00,
                                   0x00,
                                   0x14,
                                   0x63,
                                   0x00,
                                   0x00,
                                   0x00};
  std::vector<uint8_t> const m3 = {0x01, 0x01, 0x00, 0x01, 'k'};
  std::vector<uint8_t> const v3 = {0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z'};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m1.begin(), m1.end()}, {m2.begin(), m2.end()}, {m3.begin(), m3.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v1.begin(), v1.end()}, {v2.begin(), v2.end()}, {v3.begin(), v3.end()}};
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto stream = cudf::test::get_default_stream();

  // "x", "$.x", and "$['x']" must all be equivalent single-step paths.
  auto bare    = cudf::io::parquet::get_variant_field(struc, "x", stream);
  auto dollar  = cudf::io::parquet::get_variant_field(struc, "$.x", stream);
  auto bracket = cudf::io::parquet::get_variant_field(struc, "$['x']", stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *bracket);
}

TEST_F(VariantExtractTest, GetNestedPathMultiRowMixedNulls)
{
  // Row 0: { a: { b: INT32(1) } } -> path {"a","b"} = 1
  std::vector<uint8_t> const m0 = {0x01, 0x02, 0x00, 0x01, 0x02, 'a', 'b'};
  std::vector<uint8_t> const v0 = {
    0x02, 0x01, 0x00, 0x00, 0x0a, 0x02, 0x01, 0x01, 0x00, 0x05, 0x14, 0x01, 0x00, 0x00, 0x00};

  // Row 1: { a: INT32(5) } -> non-object intermediate -> null
  std::vector<uint8_t> const m1 = {0x01, 0x01, 0x00, 0x01, 'a'};
  std::vector<uint8_t> const v1 = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x05, 0x00, 0x00, 0x00};

  // Row 2: { q: INT32(7) } -> key "a" missing from dict -> null
  std::vector<uint8_t> const m2 = {0x01, 0x01, 0x00, 0x01, 'q'};
  std::vector<uint8_t> const v2 = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m0.begin(), m0.end()}, {m1.begin(), m1.end()}, {m2.begin(), m2.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v0.begin(), v0.end()}, {v1.begin(), v1.end()}, {v2.begin(), v2.end()}};
  cudf::test::structs_column_wrapper struc{{meta, val}};

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "$.a.b", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0}, {true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

// ---------------------------------------------------------------------------
// Array-indexing tests
// ---------------------------------------------------------------------------

TEST_F(VariantExtractTest, TopLevelArrayIndex)
{
  // Root value is [10, 20, 30] (INT32 array) with an empty metadata dictionary.
  auto const meta = build_metadata({});
  auto const val  = build_array_value({enc_int32(10), enc_int32(20), enc_int32(30)});
  auto struc      = wrap_single_variant(meta, val);
  auto stream     = cudf::test::get_default_stream();

  auto const i32 = cudf::data_type{cudf::type_id::INT32};
  auto first     = cudf::io::parquet::extract_variant_field(struc, "$[0]", i32, stream);
  auto third     = cudf::io::parquet::extract_variant_field(struc, "$[2]", i32, stream);
  auto oob       = cudf::io::parquet::extract_variant_field(struc, "$[5]", i32, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*first, cudf::test::fixed_width_column_wrapper<int32_t>{10});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*third, cudf::test::fixed_width_column_wrapper<int32_t>{30});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*oob,
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0}, {false}));
}

TEST_F(VariantExtractTest, ObjectArrayObject)
{
  // { "foo": [ { "bar": "ok" } ] } — dictionary ordered lex: "bar"(0), "foo"(1).
  auto const meta      = build_metadata({"bar", "foo"});
  auto const inner_ok  = enc_short_string("ok");
  auto const bar_obj   = build_single_field_object(/*fid=*/0, inner_ok);
  auto const foo_array = build_array_value({bar_obj});
  auto const outer_obj = build_single_field_object(/*fid=*/1, foo_array);
  auto struc           = wrap_single_variant(meta, outer_obj);

  auto got = cudf::io::parquet::extract_variant_field(struc,
                                                      "$.foo[0].bar",
                                                      cudf::data_type{cudf::type_id::STRING},
                                                      cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, cudf::test::strings_column_wrapper{"ok"});
}

TEST_F(VariantExtractTest, IndexOnObject)
{
  // { "foo": {} } — dict {"foo"}. Applying [0] to an object must yield null.
  auto const meta  = build_metadata({"foo"});
  auto const empty = std::vector<uint8_t>{0x02, 0x00, 0x00};  // empty object
  auto const outer = build_single_field_object(/*fid=*/0, empty);
  auto struc       = wrap_single_variant(meta, outer);

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "$.foo[0]", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0}, {false}));
}

TEST_F(VariantExtractTest, NameOnArray)
{
  // Root is [INT32(1)]; "$[0]" is an INT32 primitive; descending ".bar" into a primitive → null.
  auto const meta = build_metadata({"bar"});
  auto const val  = build_array_value({enc_int32(1)});
  auto struc      = wrap_single_variant(meta, val);

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "$[0].bar", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0}, {false}));
}

TEST_F(VariantExtractTest, NegativeIndexThrows)
{
  auto const meta = build_metadata({});
  auto const val  = build_array_value({enc_int32(10)});
  auto struc      = wrap_single_variant(meta, val);

  EXPECT_THROW(
    cudf::io::parquet::get_variant_field(struc, "$[-1]", cudf::test::get_default_stream()),
    std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Path-string / parser tests (exercised through the public API)
// ---------------------------------------------------------------------------

TEST_F(VariantExtractTest, LeadingDollarOptional)
{
  auto const meta = build_metadata({"x"});
  auto const val  = build_single_field_object(/*fid=*/0, enc_int32(42));
  auto struc      = wrap_single_variant(meta, val);
  auto stream     = cudf::test::get_default_stream();

  auto bare   = cudf::io::parquet::get_variant_field(struc, "x", stream);
  auto dollar = cudf::io::parquet::get_variant_field(struc, "$.x", stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

TEST_F(VariantExtractTest, QuotedKey)
{
  // Key literally containing a dot is only reachable through a quoted bracket step.
  auto const meta      = build_metadata({"weird.key"});
  auto const arr_val   = build_array_value({enc_int32(100), enc_int32(200), enc_int32(300)});
  auto const outer_obj = build_single_field_object(/*fid=*/0, arr_val);
  auto struc           = wrap_single_variant(meta, outer_obj);

  auto got = cudf::io::parquet::extract_variant_field(struc,
                                                      "$['weird.key'][2]",
                                                      cudf::data_type{cudf::type_id::INT32},
                                                      cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, cudf::test::fixed_width_column_wrapper<int32_t>{300});
}

TEST_F(VariantExtractTest, WildcardRejected)
{
  auto struc  = wrap_single_variant(build_metadata({}), build_array_value({enc_int32(1)}));
  auto stream = cudf::test::get_default_stream();
  // Match on the substring "[*]" in the error message.
  try {
    cudf::io::parquet::get_variant_field(struc, "$.a[*].b", stream);
    FAIL() << "expected std::invalid_argument";
  } catch (std::invalid_argument const& e) {
    EXPECT_NE(std::string_view{e.what()}.find("[*]"), std::string_view::npos);
  }
}

TEST_F(VariantExtractTest, EmptyPathRejected)
{
  auto struc  = wrap_single_variant(build_metadata({}), build_array_value({enc_int32(1)}));
  auto stream = cudf::test::get_default_stream();
  EXPECT_THROW(cudf::io::parquet::get_variant_field(struc, "", stream), std::invalid_argument);
  EXPECT_THROW(cudf::io::parquet::get_variant_field(struc, "$", stream), std::invalid_argument);
}

TEST_F(VariantExtractTest, SyntaxErrors)
{
  auto struc  = wrap_single_variant(build_metadata({}), build_array_value({enc_int32(1)}));
  auto stream = cudf::test::get_default_stream();
  for (auto const* bad : {"$..a", "$.a[", "$.a[]", "$.a.1bad", "$.", "$.'q'"}) {
    EXPECT_THROW(cudf::io::parquet::get_variant_field(struc, bad, stream), std::invalid_argument)
      << "path that should have thrown: " << bad;
  }
}

TEST_F(VariantExtractTest, BenchmarkPathSmokeCheck)
{
  // Hand-build a blob that mirrors three shapes appearing in the Python variant benchmark:
  //   { "item016": { "item017": { "item085": [ { "item018": "found" } ] } } }
  //
  // Dictionary sorted lexicographically: item016(0) < item017(1) < item018(2) < item085(3).
  auto const meta = build_metadata({"item016", "item017", "item018", "item085"});

  auto const leaf_str    = enc_short_string("found");
  auto const leaf_obj    = build_single_field_object(/*fid=*/2, leaf_str);  // {"item018":"found"}
  auto const item085_arr = build_array_value({leaf_obj});
  auto const item017_obj = build_single_field_object(/*fid=*/3, item085_arr);
  auto const item016_obj = build_single_field_object(/*fid=*/1, item017_obj);
  auto const root        = build_single_field_object(/*fid=*/0, item016_obj);
  auto struc             = wrap_single_variant(meta, root);
  auto stream            = cudf::test::get_default_stream();

  auto nested = cudf::io::parquet::extract_variant_field(
    struc, "$.item016.item017.item085[0].item018", cudf::data_type{cudf::type_id::STRING}, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*nested, cudf::test::strings_column_wrapper{"found"});

  // Purely-object prefix.
  auto raw_item017 = cudf::io::parquet::get_variant_field(struc, "$.item016.item017", stream);
  EXPECT_EQ(raw_item017->type().id(), cudf::type_id::STRUCT);

  // Bare-name first step equivalent to `$.item016`.
  auto bare   = cudf::io::parquet::get_variant_field(struc, "item016", stream);
  auto dollar = cudf::io::parquet::get_variant_field(struc, "$.item016", stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

TEST_F(VariantExtractTest, CastInt32Composed)
{
  // { "arr": [10, 20, 30] } — extract element index 1 as INT32.
  auto const meta      = build_metadata({"arr"});
  auto const arr       = build_array_value({enc_int32(10), enc_int32(20), enc_int32(30)});
  auto const outer_obj = build_single_field_object(/*fid=*/0, arr);
  auto struc           = wrap_single_variant(meta, outer_obj);

  auto got = cudf::io::parquet::extract_variant_field(
    struc, "$.arr[1]", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, cudf::test::fixed_width_column_wrapper<int32_t>{20});
}
