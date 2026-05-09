/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apache_variant_fixtures.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/utilities/span.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

struct VariantExtractTest : public cudf::test::BaseFixture {};

TEST_F(VariantExtractTest, NullStructRow)
{
  cudf::test::lists_column_wrapper<uint8_t> meta{
    {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')},
    {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00},
    {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00}};
  cudf::test::structs_column_wrapper struc{{meta, val}, std::vector<bool>{true, false}};

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 0}, {true, false});
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
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
  auto x      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp({7, 42, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);

  auto k = cudf::io::parquet::experimental::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper k_exp({"hi", "", "zzz"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*k, k_exp);

  auto y = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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
  auto x      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);
  auto k = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::get_variant_field(
    struc, "x", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->num_children(), 2);
  EXPECT_EQ(got->size(), 1);

  auto const child0 = got->view().child(0);
  auto const child1 = got->view().child(1);
  EXPECT_EQ(child0.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(child1.type().id(), cudf::type_id::LIST);

  // The extracted value for "x" should be the INT32 encoding: {0x14, 0x07, 0x00, 0x00, 0x00}
  // Verify it can be decoded via cast_variant
  auto casted = cudf::io::parquet::experimental::cast_variant(
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

  auto got = cudf::io::parquet::experimental::get_variant_field(
    struc, "missing", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(got->null_count(), 1);

  // Metadata child should still be valid
  auto const meta_child = got->view().child(0);
  EXPECT_EQ(meta_child.type().id(), cudf::type_id::LIST);
  EXPECT_EQ(meta_child.size(), 1);
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
  auto extract_x = cudf::io::parquet::experimental::extract_variant_field(
    struc, "x", cudf::data_type{cudf::type_id::INT32}, stream);

  // get_variant_field + cast_variant (two-step)
  auto intermediate = cudf::io::parquet::experimental::get_variant_field(struc, "x", stream);
  auto two_step_x   = cudf::io::parquet::experimental::cast_variant(
    intermediate->view(), cudf::data_type{cudf::type_id::INT32}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_x, *two_step_x);

  // Same for string field "k"
  auto extract_k = cudf::io::parquet::experimental::extract_variant_field(
    struc, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  auto intermediate_k = cudf::io::parquet::experimental::get_variant_field(struc, "k", stream);
  auto two_step_k     = cudf::io::parquet::experimental::cast_variant(
    intermediate_k->view(), cudf::data_type{cudf::type_id::STRING}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_k, *two_step_k);
}

namespace {

// Build a single-row VARIANT struct column from a (metadata, value) byte pair.
template <std::size_t M, std::size_t V>
cudf::test::structs_column_wrapper make_apache_variant(
  cudf::test::apache_variant_fixtures::fixture<M, V> const& f)
{
  cudf::test::lists_column_wrapper<uint8_t> m(f.metadata.begin(), f.metadata.end());
  cudf::test::lists_column_wrapper<uint8_t> v(f.value.begin(), f.value.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

}  // namespace

TEST_F(VariantExtractTest, ApachePrimitiveInt8)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::primitive_int8);
  auto got      = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{42}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApachePrimitiveInt16)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::primitive_int16);
  auto got      = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT16}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int16_t> expected{int16_t{1234}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApachePrimitiveInt32)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::primitive_int32);

  auto got = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected{123456};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApachePrimitiveInt64)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::primitive_int64);
  auto got      = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::INT64}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int64_t> expected{int64_t{1234567890123456789LL}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheShortString)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::short_string);

  auto got = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  // Decoded from short_string.value: skip the 1-byte header, take the rest.
  std::string const expected_str(reinterpret_cast<char const*>(afv::short_string.value.data() + 1),
                                 afv::short_string.value.size() - 1);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheLongString)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::primitive_string);

  auto got = cudf::io::parquet::experimental::cast_variant(
    struc, cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  // Long-string layout: 1 header byte + 4-byte LE length + payload.
  std::string const expected_str(
    reinterpret_cast<char const*>(afv::primitive_string.value.data() + 5),
    afv::primitive_string.value.size() - 5);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitiveExtractString)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_primitive);

  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "string_field",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"Apache Parquet"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheNestedGetVariantField)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_nested);
  auto stream   = cudf::test::get_default_stream();

  auto species = cudf::io::parquet::experimental::get_variant_field(struc, "species", stream);
  EXPECT_EQ(species->type().id(), cudf::type_id::STRUCT);

  auto name = cudf::io::parquet::experimental::get_variant_field(species->view(), "name", stream);
  EXPECT_EQ(name->type().id(), cudf::type_id::STRUCT);

  auto name_str = cudf::io::parquet::experimental::cast_variant(
    name->view(), cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"lava monster"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*name_str, expected);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitive_int_field_asInt8)
{
  // int_field is encoded as INT8 (header 0x0c). cast_variant(INT8) -> 1.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_primitive);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "int_field", cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitive_timestamp_field_asString)
{
  // timestamp_field is encoded as a long string ("2025-04-16T12:34:56.78"), so STRING decodes.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_primitive);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "timestamp_field",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"2025-04-16T12:34:56.78"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitive_missing_field_isNull)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_primitive);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "no_such_field",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(VariantExtractTest, ApacheObjectPrimitive_null_field_isNull)
{
  // null_field is the Variant null primitive; cast to STRING yields null.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_primitive);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "null_field", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(VariantExtractTest, ApacheObjectNested_observation_location)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_nested);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "$.observation.location",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"In the Volcano"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectNested_observation_time)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_nested);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "$.observation.time",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"12:34:56"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectNested_species_population_asInt16)
{
  // population=6789 is encoded as INT16. cast_variant(INT16) -> 6789.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_nested);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "$.species.population",
                                                           cudf::data_type{cudf::type_id::INT16},
                                                           cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int16_t> expected{int16_t{6789}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheObjectNested_id_asInt8)
{
  // id=1 is encoded as INT8 (0x0c 0x01). cast_variant(INT8) -> 1.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_nested);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$.id", cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheArrayPrimitive_indexZero_asInt8)
{
  // Element 0 is INT8(2). cast_variant(INT8) -> 2.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_primitive);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[0]", cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{2}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheArrayPrimitive_outOfBounds_isNull)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_primitive);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[42]", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(VariantExtractTest, ApacheArrayNested_index2_type)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_nested);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[2].type", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"if"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheArrayNested_index0_thing_names_index1)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_nested);
  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "$[0].thing.names[1]",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected({"Spider"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, ApacheArrayNested_indexNull_isNull)
{
  // Element 1 is the Variant null primitive. Walking into it returns null.
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_nested);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[1].id", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(VariantExtractTest, ApacheObjectEmpty_anyKey_isNull)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::object_empty);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "foo", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(VariantExtractTest, ApacheArrayEmpty_anyIndex_isNull)
{
  namespace afv = cudf::test::apache_variant_fixtures;
  auto struc    = make_apache_variant(afv::array_empty);
  auto got      = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[0]", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
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

// Shared fixture: Apache parquet-testing object_nested blob. See
// apache_variant_fixtures.hpp for the underlying bytes and provenance.
cudf::test::structs_column_wrapper make_apache_nested_col()
{
  return make_apache_variant(cudf::test::apache_variant_fixtures::object_nested);
}

}  // namespace

TEST_F(VariantExtractTest, GetNestedPathSpeciesName)
{
  auto col    = make_apache_nested_col();
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.species.name", cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"lava monster"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);

  // Parity: single-call path equals chained calls.
  auto chained_species = cudf::io::parquet::experimental::get_variant_field(col, "species", stream);
  auto chained_name =
    cudf::io::parquet::experimental::get_variant_field(chained_species->view(), "name", stream);
  auto chained_str = cudf::io::parquet::experimental::cast_variant(
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
  auto single = cudf::io::parquet::experimental::get_variant_field(
    col, "$.observation.value.temperature", stream);

  auto obs  = cudf::io::parquet::experimental::get_variant_field(col, "observation", stream);
  auto vobj = cudf::io::parquet::experimental::get_variant_field(obs->view(), "value", stream);
  auto chained =
    cudf::io::parquet::experimental::get_variant_field(vobj->view(), "temperature", stream);

  // Both columns must be VARIANT structs with the same value child contents.
  EXPECT_EQ(single->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(chained->type().id(), cudf::type_id::STRUCT);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(single->view().child(1), chained->view().child(1));
}

TEST_F(VariantExtractTest, GetNestedPathMissingIntermediate)
{
  auto col    = make_apache_nested_col();
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$.a.b", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(VariantExtractTest, GetNestedPathEmptyThrows)
{
  auto col = make_apache_nested_col();
  EXPECT_THROW(
    cudf::io::parquet::experimental::get_variant_field(col, "", cudf::test::get_default_stream()),
    std::invalid_argument);
  EXPECT_THROW(cudf::io::parquet::experimental::extract_variant_field(
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
  auto bare    = cudf::io::parquet::experimental::get_variant_field(struc, "x", stream);
  auto dollar  = cudf::io::parquet::experimental::get_variant_field(struc, "$.x", stream);
  auto bracket = cudf::io::parquet::experimental::get_variant_field(struc, "$['x']", stream);

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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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
  auto first = cudf::io::parquet::experimental::extract_variant_field(struc, "$[0]", i32, stream);
  auto third = cudf::io::parquet::experimental::extract_variant_field(struc, "$[2]", i32, stream);
  auto oob   = cudf::io::parquet::experimental::extract_variant_field(struc, "$[5]", i32, stream);

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

  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
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

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$[0].bar", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0}, {false}));
}

TEST_F(VariantExtractTest, NegativeIndexThrows)
{
  auto const meta = build_metadata({});
  auto const val  = build_array_value({enc_int32(10)});
  auto struc      = wrap_single_variant(meta, val);

  EXPECT_THROW(cudf::io::parquet::experimental::get_variant_field(
                 struc, "$[-1]", cudf::test::get_default_stream()),
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

  auto bare   = cudf::io::parquet::experimental::get_variant_field(struc, "x", stream);
  auto dollar = cudf::io::parquet::experimental::get_variant_field(struc, "$.x", stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

TEST_F(VariantExtractTest, QuotedKey)
{
  // Key literally containing a dot is only reachable through a quoted bracket step.
  auto const meta      = build_metadata({"weird.key"});
  auto const arr_val   = build_array_value({enc_int32(100), enc_int32(200), enc_int32(300)});
  auto const outer_obj = build_single_field_object(/*fid=*/0, arr_val);
  auto struc           = wrap_single_variant(meta, outer_obj);

  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
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
    cudf::io::parquet::experimental::get_variant_field(struc, "$.a[*].b", stream);
    FAIL() << "expected std::invalid_argument";
  } catch (std::invalid_argument const& e) {
    EXPECT_NE(std::string_view{e.what()}.find("[*]"), std::string_view::npos);
  }
}

TEST_F(VariantExtractTest, EmptyPathRejected)
{
  auto struc  = wrap_single_variant(build_metadata({}), build_array_value({enc_int32(1)}));
  auto stream = cudf::test::get_default_stream();
  EXPECT_THROW(cudf::io::parquet::experimental::get_variant_field(struc, "", stream),
               std::invalid_argument);
  EXPECT_THROW(cudf::io::parquet::experimental::get_variant_field(struc, "$", stream),
               std::invalid_argument);
}

TEST_F(VariantExtractTest, SyntaxErrors)
{
  auto struc  = wrap_single_variant(build_metadata({}), build_array_value({enc_int32(1)}));
  auto stream = cudf::test::get_default_stream();
  for (auto const* bad : {"$..a", "$.a[", "$.a[]", "$.a.1bad", "$.", "$.'q'"}) {
    EXPECT_THROW(cudf::io::parquet::experimental::get_variant_field(struc, bad, stream),
                 std::invalid_argument)
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

  auto nested = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$.item016.item017.item085[0].item018", cudf::data_type{cudf::type_id::STRING}, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*nested, cudf::test::strings_column_wrapper{"found"});

  // Purely-object prefix.
  auto raw_item017 =
    cudf::io::parquet::experimental::get_variant_field(struc, "$.item016.item017", stream);
  EXPECT_EQ(raw_item017->type().id(), cudf::type_id::STRUCT);

  // Bare-name first step equivalent to `$.item016`.
  auto bare   = cudf::io::parquet::experimental::get_variant_field(struc, "item016", stream);
  auto dollar = cudf::io::parquet::experimental::get_variant_field(struc, "$.item016", stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

TEST_F(VariantExtractTest, CastInt32Composed)
{
  // { "arr": [10, 20, 30] } — extract element index 1 as INT32.
  auto const meta      = build_metadata({"arr"});
  auto const arr       = build_array_value({enc_int32(10), enc_int32(20), enc_int32(30)});
  auto const outer_obj = build_single_field_object(/*fid=*/0, arr);
  auto struc           = wrap_single_variant(meta, outer_obj);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    struc, "$.arr[1]", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, cudf::test::fixed_width_column_wrapper<int32_t>{20});
}

// ---------------------------------------------------------------------------
// Scale tests — dictionary / object scans an order of magnitude larger than
// any other test in this file, and a multi-row deep-path fixture large enough
// to exercise the atomic null-mask updates in the sizes/copy kernels.
// ---------------------------------------------------------------------------

namespace {

// Build a VARIANT object blob with `n_fields` fields.  Field ids are 0..n_fields-1
// (sorted as required by the spec) and each field holds a bare INT32 equal to its
// field id.  Uses 1-byte field_id_size and 1-byte field_off_size; n_fields must be
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

// Sorted dictionary of N zero-padded two-digit keys "k<NN>".
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

}  // namespace

TEST_F(VariantExtractTest, LargeDictionaryAndObjectScan)
{
  // 50-entry dictionary + 50-field object — >10x the size of any other test in this
  // file.  Exercises the scan loops in device_find_key_in_metadata and
  // device_locate_object_field at a depth that would catch regressions in their
  // scan-bound arithmetic without requiring a workload-scale fixture.
  auto const keys = make_numeric_keys(50);
  auto const meta = build_metadata(keys);
  auto const val  = build_sequential_int32_object(50);
  auto struc      = wrap_single_variant(meta, val);
  auto stream     = cudf::test::get_default_stream();
  auto const i32  = cudf::data_type{cudf::type_id::INT32};

  // First, middle, and last keys each decode to their own field id.
  auto first = cudf::io::parquet::experimental::extract_variant_field(struc, "k00", i32, stream);
  auto mid   = cudf::io::parquet::experimental::extract_variant_field(struc, "k24", i32, stream);
  auto last  = cudf::io::parquet::experimental::extract_variant_field(struc, "k49", i32, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*first, cudf::test::fixed_width_column_wrapper<int32_t>{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*mid, cudf::test::fixed_width_column_wrapper<int32_t>{24});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*last, cudf::test::fixed_width_column_wrapper<int32_t>{49});

  // Not-present key must still yield null after a full dictionary scan.
  auto missing = cudf::io::parquet::experimental::extract_variant_field(struc, "k99", i32, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*missing,
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0}, {false}));
}

TEST_F(VariantExtractTest, DeepMixedPathManyRowsWithNulls)
{
  // 128 rows of a depth-5 mixed-step path $.a.b[0].c.d, cycling through four row
  // shapes that null at different depths.  With 128 rows spread across 4 bitmask
  // words and ~75% of rows nulling, this stresses the per-row null-mask updates in
  // the sizes and copy kernels — threads within a warp all share one bitmask word,
  // so non-atomic bit clears would race and silently drop some nulls.  A smaller
  // fixture (e.g. 3 rows) would not reliably catch that.
  std::vector<std::string> const dict = {"a", "b", "c", "d"};  // fids: a=0,b=1,c=2,d=3
  auto const meta                     = build_metadata(dict);

  // Shape 0: intact — {a:{b:[{c:{d:"leaf"}}]}}
  auto const s0_d    = enc_short_string("leaf");
  auto const s0_cd   = build_single_field_object(/*fid=d*/ 3, s0_d);
  auto const s0_elem = build_single_field_object(/*fid=c*/ 2, s0_cd);
  auto const s0_arr  = build_array_value({s0_elem});
  auto const s0_ab   = build_single_field_object(/*fid=b*/ 1, s0_arr);
  auto const s0_root = build_single_field_object(/*fid=a*/ 0, s0_ab);

  // Shape 1: missing key at depth 1 — root has "b" but no "a".
  auto const s1_b    = enc_int32(0);
  auto const s1_root = build_single_field_object(/*fid=b*/ 1, s1_b);

  // Shape 2: kind mismatch at depth 3 — {a:{b:INT32(7)}} makes [0] nonsensical.
  auto const s2_bval = enc_int32(7);
  auto const s2_ab   = build_single_field_object(/*fid=b*/ 1, s2_bval);
  auto const s2_root = build_single_field_object(/*fid=a*/ 0, s2_ab);

  // Shape 3: array OOB at depth 3 — {a:{b:[]}} makes [0] out of bounds.
  auto const s3_arr  = build_array_value({});
  auto const s3_ab   = build_single_field_object(/*fid=b*/ 1, s3_arr);
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

  auto struc = wrap_multi_row_variant(meta_rows, val_rows);

  auto got =
    cudf::io::parquet::experimental::extract_variant_field(struc,
                                                           "$.a.b[0].c.d",
                                                           cudf::data_type{cudf::type_id::STRING},
                                                           cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected(exp_strs.begin(), exp_strs.end(), exp_valid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, expected);
}
