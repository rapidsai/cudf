/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/variant.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <filesystem>
#include <string>

namespace {

[[nodiscard]] std::string variant_fixture_path(std::string const& filename)
{
#ifndef VARIANT_PARQUET_FIXTURE_DIR
  CUDF_FAIL("VARIANT_PARQUET_FIXTURE_DIR not defined");
#endif
  return std::string(VARIANT_PARQUET_FIXTURE_DIR) + "/" + filename;
}

struct ParquetVariantRoundtripTest : public cudf::test::BaseFixture {};

}  // namespace

TEST_F(ParquetVariantRoundtripTest, ReadMinimalVariantParquet)
{
  auto const path = variant_fixture_path("variant_minimal.parquet");
  ASSERT_TRUE(std::filesystem::exists(path)) << path;

  cudf::io::parquet_reader_options opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{path});
  auto const result = cudf::io::read_parquet(opts);
  ASSERT_EQ(result.tbl->num_columns(), 1);
  auto const col = result.tbl->get_column(0);
  ASSERT_EQ(col.type().id(), cudf::type_id::STRUCT);
  cudf::structs_column_view const scv{col};
  ASSERT_EQ(scv.num_children(), 2);
  cudf::lists_column_view const meta{scv.child(0)};
  cudf::lists_column_view const val{scv.child(1)};
  ASSERT_EQ(meta.child().type().id(), cudf::type_id::UINT8);
  ASSERT_EQ(val.child().type().id(), cudf::type_id::UINT8);

  auto got = cudf::io::parquet::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int32_t> expected({7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ParquetVariantRoundtripTest, ReadMultirowVariantParquet)
{
  auto const path = variant_fixture_path("variant_multirow.parquet");
  ASSERT_TRUE(std::filesystem::exists(path)) << path;

  cudf::io::parquet_reader_options opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{path});
  auto const result = cudf::io::read_parquet(opts);
  ASSERT_EQ(result.tbl->num_columns(), 1);
  auto const col = result.tbl->get_column(0);
  ASSERT_EQ(col.size(), 3);

  auto stream = cudf::test::get_default_stream();

  auto x = cudf::io::parquet::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp({7, 42, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);

  auto k = cudf::io::parquet::extract_variant_field(
    col, "k", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper k_exp({"hi", "", "zzz"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*k, k_exp);

  auto y = cudf::io::parquet::extract_variant_field(
    col, "y", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> y_exp({0, 99, 0}, {false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*y, y_exp);
}
