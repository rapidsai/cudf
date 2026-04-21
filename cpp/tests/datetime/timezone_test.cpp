/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/table/table.hpp>
#include <cudf/timezone.hpp>
#include <cudf/utilities/error.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

constexpr char const* kCanonicalZonePath = "/usr/share/zoneinfo/America/Los_Angeles";

}  // namespace

class TimezoneTransitionTableTest : public cudf::test::BaseFixture {};

TEST_F(TimezoneTransitionTableTest, UtcShortCircuitsWithoutReadingFile)
{
  auto const table = cudf::make_timezone_transition_table(std::nullopt, "UTC");
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 0);
}

TEST_F(TimezoneTransitionTableTest, EmptyZoneNameShortCircuitsWithoutReadingFile)
{
  auto const table = cudf::make_timezone_transition_table(std::nullopt, "");
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 0);
}

TEST_F(TimezoneTransitionTableTest, CanonicalZoneProducesTwoColumnTable)
{
  namespace fs = std::filesystem;
  if (!fs::exists(kCanonicalZonePath)) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }

  auto const table =
    cudf::make_timezone_transition_table("/usr/share/zoneinfo", "America/Los_Angeles");
  ASSERT_EQ(table->num_columns(), 2);
  // Sanity: the future cycle dominates the row count, so we expect hundreds of rows.
  EXPECT_GT(table->num_rows(), 100);
}

TEST_F(TimezoneTransitionTableTest, UnknownZoneThrows)
{
  namespace fs = std::filesystem;
  if (!fs::exists(kCanonicalZonePath)) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }

  EXPECT_THROW(cudf::make_timezone_transition_table("/usr/share/zoneinfo", "Not_A/Real_Zone_bXYZ"),
               cudf::logic_error);
}

class TimezoneAliasResolutionTest : public cudf::test::BaseFixture {
 protected:
  void SetUp() override
  {
    auto const base = std::filesystem::temp_directory_path();
    // Include the test name so parallel cases don't collide.
    auto const test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tz_dir_              = base / (std::string{"cudf_tz_alias_test_"} + test_name);
    std::filesystem::remove_all(tz_dir_);
    std::filesystem::create_directories(tz_dir_);
  }

  void TearDown() override
  {
    std::error_code ec;
    std::filesystem::remove_all(tz_dir_, ec);
  }

  [[nodiscard]] bool install_zone(std::string_view zone_name) const
  {
    std::error_code ec;
    if (!std::filesystem::exists(kCanonicalZonePath, ec)) { return false; }
    auto const dst = tz_dir_ / zone_name;
    std::filesystem::create_directories(dst.parent_path(), ec);
    std::filesystem::copy_file(
      kCanonicalZonePath, dst, std::filesystem::copy_options::overwrite_existing, ec);
    return !ec;
  }

  void write_tzdata_zi(std::string_view contents) const
  {
    std::ofstream{tz_dir_ / "tzdata.zi"} << contents;
  }

  [[nodiscard]] std::string dir() const { return tz_dir_.string(); }

 private:
  std::filesystem::path tz_dir_;
};

TEST_F(TimezoneAliasResolutionTest, DirectLookupUnaffectedByNewFallback)
{
  if (!install_zone("America/Los_Angeles")) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }

  auto const table = cudf::make_timezone_transition_table(dir(), "America/Los_Angeles");
  EXPECT_GT(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(TimezoneAliasResolutionTest, ResolvesShortFormLinkFromTzdataZi)
{
  if (!install_zone("America/Los_Angeles")) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }
  write_tzdata_zi(
    "# synthetic tzdata.zi for libcudf tests\n"
    "L America/Los_Angeles US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), "America/Los_Angeles");
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ResolvesLongFormLinkFromTzdataZi)
{
  if (!install_zone("America/Los_Angeles")) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }
  write_tzdata_zi("Link America/Los_Angeles US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), "America/Los_Angeles");
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ResolvesChainedLinks)
{
  if (!install_zone("America/Los_Angeles")) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }
  // Neither "intermediate" nor "US/Pacific" exists on disk. The resolver must traverse
  //   US/Pacific -> intermediate -> America/Los_Angeles to find a real file.
  write_tzdata_zi(
    "L America/Los_Angeles intermediate\n"
    "L intermediate US/Pacific\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), "America/Los_Angeles");
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

TEST_F(TimezoneAliasResolutionTest, ThrowsWhenLinkTargetIsAlsoMissing)
{
  // No zone files installed in tz_dir_.
  write_tzdata_zi("L Also/Missing US/Pacific\n");

  EXPECT_THROW(cudf::make_timezone_transition_table(dir(), "US/Pacific"), cudf::logic_error);
}

TEST_F(TimezoneAliasResolutionTest, ThrowsWhenNoTzdataZiPresent)
{
  EXPECT_THROW(cudf::make_timezone_transition_table(dir(), "US/Pacific"), cudf::logic_error);
}

TEST_F(TimezoneAliasResolutionTest, IgnoresCommentsAndNonLinkDirectives)
{
  if (!install_zone("America/Los_Angeles")) {
    GTEST_SKIP() << "System is missing " << kCanonicalZonePath;
  }
  write_tzdata_zi(
    "# a leading comment\n"
    "\n"
    "R SomeRule 1970 o - Jan 1 0 0 S\n"      // `Rule` entry, must be ignored
    "Z Fake/Zone 0 - LMT\n"                  // `Zone` entry, must be ignored
    "   L America/Los_Angeles US/Pacific\n"  // link with leading whitespace
    "# trailing comment\n");

  auto const via_canonical = cudf::make_timezone_transition_table(dir(), "America/Los_Angeles");
  auto const via_alias     = cudf::make_timezone_transition_table(dir(), "US/Pacific");
  CUDF_TEST_EXPECT_TABLES_EQUAL(via_canonical->view(), via_alias->view());
}

CUDF_TEST_PROGRAM_MAIN()
