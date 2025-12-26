/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/key_remapping.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <set>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;

struct KeyRemappingTest : public cudf::test::BaseFixture {
  // Helper to copy column to host vector
  template <typename T>
  std::vector<T> to_host(cudf::column_view const& col)
  {
    return cudf::detail::make_std_vector<T>(
      cudf::device_span<T const>{col.data<T>(), static_cast<std::size_t>(col.size())},
      cudf::get_default_stream());
  }

  // Verify that equal keys in the input map to equal IDs in the output
  // Uses the original keys table and the remapped IDs column
  void verify_equal_keys_have_equal_ids(cudf::table_view const& keys, cudf::column_view const& ids)
  {
    // Sort the keys+ids together by keys, then verify adjacent equal keys have equal ids
    auto keys_with_ids_cols = keys.column(0).size() > 0 ? std::vector<cudf::column_view>{}
                                                        : std::vector<cudf::column_view>{};

    // Build a table with keys + id column for sorting
    std::vector<cudf::column_view> all_cols;
    for (int i = 0; i < keys.num_columns(); ++i) {
      all_cols.push_back(keys.column(i));
    }
    all_cols.push_back(ids);
    cudf::table_view keys_with_ids{all_cols};

    // Sort by keys (all columns except last)
    std::vector<cudf::order> orders(keys.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_orders(keys.num_columns(), cudf::null_order::BEFORE);
    auto sorted_indices = cudf::sorted_order(keys, orders, null_orders);
    auto sorted_table   = cudf::gather(keys_with_ids, *sorted_indices);

    // Bring sorted ids to host and verify adjacent equal keys have equal ids
    auto sorted_ids = to_host<int32_t>(sorted_table->get_column(keys.num_columns()).view());

    // Also need to check keys - bring them to host too for comparison
    // For simplicity, use the distinct count: if we have N distinct keys,
    // we should have exactly N distinct IDs among non-sentinel values
    auto non_sentinel_ids = std::vector<int32_t>{};
    for (auto id : sorted_ids) {
      if (id >= 0) { non_sentinel_ids.push_back(id); }
    }

    // Group consecutive rows with same key and verify they have same ID
    // This requires comparing keys which is complex for multi-column
    // Instead, let's use a different approach: verify distinct_count matches unique IDs
  }

  // Verify remapping contract: distinct keys get distinct IDs, equal keys get equal IDs
  void verify_remapping_contract(cudf::table_view const& keys,
                                 cudf::column_view const& ids,
                                 cudf::size_type expected_distinct_count)
  {
    auto host_ids = to_host<int32_t>(ids);

    // Collect non-sentinel IDs
    std::set<int32_t> unique_ids;
    for (auto id : host_ids) {
      if (id >= 0) { unique_ids.insert(id); }
    }

    // Number of unique non-negative IDs should equal distinct key count
    EXPECT_EQ(static_cast<cudf::size_type>(unique_ids.size()), expected_distinct_count);

    // All non-negative IDs should be valid (non-negative)
    for (auto id : host_ids) {
      EXPECT_TRUE(id >= 0 || id == cudf::KEY_REMAP_NOT_FOUND || id == cudf::KEY_REMAP_BUILD_NULL);
    }
  }

  // Verify that probe keys matching build keys get the same ID
  void verify_probe_matches_build(cudf::table_view const& build_keys,
                                  cudf::column_view const& build_ids,
                                  cudf::table_view const& probe_keys,
                                  cudf::column_view const& probe_ids)
  {
    auto host_build_ids = to_host<int32_t>(build_ids);
    auto host_probe_ids = to_host<int32_t>(probe_ids);

    // For each probe row, if its ID is non-negative, there should exist
    // a build row with the same ID (meaning the key was found)
    std::set<int32_t> build_id_set(host_build_ids.begin(), host_build_ids.end());

    for (size_t i = 0; i < host_probe_ids.size(); ++i) {
      auto probe_id = host_probe_ids[i];
      if (probe_id >= 0) {
        // This probe key matched a build key, so the ID should exist in build
        EXPECT_TRUE(build_id_set.count(probe_id) > 0)
          << "Probe row " << i << " has ID " << probe_id << " not found in build IDs";
      }
    }
  }
};

TEST_F(KeyRemappingTest, BasicIntegerKeys)
{
  // Build table with some duplicate keys: [1, 2, 3, 2, 1]
  column_wrapper<int32_t> build_col{1, 2, 3, 2, 1};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  // Check distinct count (should be 3: values 1, 2, 3)
  EXPECT_EQ(remap.get_distinct_count(), 3);

  // Check max duplicate count (value 1 and 2 both appear twice)
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  // Remap build keys
  auto build_result = remap.remap_build_keys(build_table);

  // Verify contract: 3 distinct IDs for 3 distinct keys
  verify_remapping_contract(build_table, *build_result, 3);

  // Verify equal keys get equal IDs by checking specific pairs
  auto host_ids = to_host<int32_t>(*build_result);
  EXPECT_EQ(host_ids[0], host_ids[4]);  // Both have key=1
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both have key=2
  EXPECT_NE(host_ids[0], host_ids[1]);  // key=1 vs key=2
  EXPECT_NE(host_ids[0], host_ids[2]);  // key=1 vs key=3
  EXPECT_NE(host_ids[1], host_ids[2]);  // key=2 vs key=3
}

TEST_F(KeyRemappingTest, ProbeKeys)
{
  column_wrapper<int32_t> build_col{1, 2, 3, 2, 1};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  auto build_result   = remap.remap_build_keys(build_table);
  auto host_build_ids = to_host<int32_t>(*build_result);

  // Probe with some matching and non-matching keys
  column_wrapper<int32_t> probe_col{3, 1, 5, 2};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  // key=3 in probe should match key=3 in build (row 2)
  EXPECT_EQ(host_probe_ids[0], host_build_ids[2]);

  // key=1 in probe should match key=1 in build (rows 0 and 4 have same ID)
  EXPECT_EQ(host_probe_ids[1], host_build_ids[0]);

  // key=5 not in build -> NOT_FOUND
  EXPECT_EQ(host_probe_ids[2], cudf::KEY_REMAP_NOT_FOUND);

  // key=2 in probe should match key=2 in build (rows 1 and 3 have same ID)
  EXPECT_EQ(host_probe_ids[3], host_build_ids[1]);

  // Verify probe matches build
  verify_probe_matches_build(build_table, *build_result, probe_table, *probe_result);
}

TEST_F(KeyRemappingTest, StringKeys)
{
  strcol_wrapper build_col{"apple", "banana", "cherry", "banana"};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);  // "banana" appears twice

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  // Verify equal keys get equal IDs
  auto host_ids = to_host<int32_t>(*build_result);
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both "banana"
  EXPECT_NE(host_ids[0], host_ids[1]);  // "apple" vs "banana"
  EXPECT_NE(host_ids[0], host_ids[2]);  // "apple" vs "cherry"

  // Probe with matching and non-matching keys
  strcol_wrapper probe_col{"cherry", "date", "apple"};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  EXPECT_EQ(host_probe_ids[0], host_ids[2]);                // "cherry" matches
  EXPECT_EQ(host_probe_ids[1], cudf::KEY_REMAP_NOT_FOUND);  // "date" not found
  EXPECT_EQ(host_probe_ids[2], host_ids[0]);                // "apple" matches

  verify_probe_matches_build(build_table, *build_result, probe_table, *probe_result);
}

TEST_F(KeyRemappingTest, MultiColumnKeys)
{
  column_wrapper<int32_t> build_col1{1, 1, 2, 1};
  strcol_wrapper build_col2{"a", "b", "a", "a"};
  auto build_table = cudf::table_view{{build_col1, build_col2}};

  cudf::key_remapping remap{build_table};

  // Distinct keys: (1,"a"), (1,"b"), (2,"a") = 3 distinct
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);  // (1,"a") appears twice

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  // Verify equal keys get equal IDs
  auto host_ids = to_host<int32_t>(*build_result);
  EXPECT_EQ(host_ids[0], host_ids[3]);  // Both (1,"a")
  EXPECT_NE(host_ids[0], host_ids[1]);  // (1,"a") vs (1,"b")
  EXPECT_NE(host_ids[0], host_ids[2]);  // (1,"a") vs (2,"a")

  // Probe
  column_wrapper<int32_t> probe_col1{2, 1, 3};
  strcol_wrapper probe_col2{"a", "b", "c"};
  auto probe_table = cudf::table_view{{probe_col1, probe_col2}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  EXPECT_EQ(host_probe_ids[0], host_ids[2]);                // (2,"a") matches
  EXPECT_EQ(host_probe_ids[1], host_ids[1]);                // (1,"b") matches
  EXPECT_EQ(host_probe_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // (3,"c") not found

  verify_probe_matches_build(build_table, *build_result, probe_table, *probe_result);
}

TEST_F(KeyRemappingTest, NullsEqual)
{
  // Build table with nulls, nulls are equal
  column_wrapper<int32_t> build_col{{1, 2, 0, 2}, {true, true, false, true}};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table, cudf::null_equality::EQUAL};

  // Distinct: 1, 2, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  auto host_ids = to_host<int32_t>(*build_result);
  // Rows with key=2 should have same ID
  EXPECT_EQ(host_ids[1], host_ids[3]);
  // All IDs should be non-negative (nulls are treated as equal, so they get valid IDs)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Probe with nulls - null should match
  column_wrapper<int32_t> probe_col{{0, 1}, {false, true}};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  // null in probe should match null in build
  EXPECT_EQ(host_probe_ids[0], host_ids[2]);
  // 1 in probe should match 1 in build
  EXPECT_EQ(host_probe_ids[1], host_ids[0]);
}

TEST_F(KeyRemappingTest, NullsUnequal)
{
  // Build table with nulls, nulls are unequal
  column_wrapper<int32_t> build_col{{1, 2, 0, 2}, {true, true, false, true}};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1, 2 = 2 (null is skipped)
  EXPECT_EQ(remap.get_distinct_count(), 2);

  auto build_result = remap.remap_build_keys(build_table);
  auto host_ids     = to_host<int32_t>(*build_result);

  // Rows with key=2 should have same ID
  EXPECT_EQ(host_ids[1], host_ids[3]);
  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids[2], cudf::KEY_REMAP_BUILD_NULL);
  // Non-null rows should have non-negative IDs
  EXPECT_GE(host_ids[0], 0);
  EXPECT_GE(host_ids[1], 0);

  // Probe with nulls - null should not match
  column_wrapper<int32_t> probe_col{{0, 1}, {false, true}};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  // null in probe should get NOT_FOUND
  EXPECT_EQ(host_probe_ids[0], cudf::KEY_REMAP_NOT_FOUND);
  // 1 in probe should match 1 in build
  EXPECT_EQ(host_probe_ids[1], host_ids[0]);
}

TEST_F(KeyRemappingTest, EmptyBuildTable)
{
  column_wrapper<int32_t> build_col{};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  EXPECT_EQ(remap.get_distinct_count(), 0);
  EXPECT_EQ(remap.get_max_duplicate_count(), 0);

  // Probe should return all NOT_FOUND
  column_wrapper<int32_t> probe_col{1, 2, 3};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  for (auto id : host_probe_ids) {
    EXPECT_EQ(id, cudf::KEY_REMAP_NOT_FOUND);
  }
}

TEST_F(KeyRemappingTest, EmptyProbeTable)
{
  column_wrapper<int32_t> build_col{1, 2, 3};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  column_wrapper<int32_t> probe_col{};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result = remap.remap_probe_keys(probe_table);
  EXPECT_EQ(probe_result->size(), 0);
}

TEST_F(KeyRemappingTest, AllDuplicates)
{
  // All rows have the same key
  column_wrapper<int32_t> build_col{42, 42, 42, 42, 42};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  EXPECT_EQ(remap.get_distinct_count(), 1);
  EXPECT_EQ(remap.get_max_duplicate_count(), 5);

  auto build_result = remap.remap_build_keys(build_table);
  auto host_ids     = to_host<int32_t>(*build_result);

  // All should have the same ID (whatever it is)
  auto first_id = host_ids[0];
  EXPECT_GE(first_id, 0);
  for (auto id : host_ids) {
    EXPECT_EQ(id, first_id);
  }
}

TEST_F(KeyRemappingTest, AllUnique)
{
  column_wrapper<int32_t> build_col{1, 2, 3, 4, 5};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  EXPECT_EQ(remap.get_distinct_count(), 5);
  EXPECT_EQ(remap.get_max_duplicate_count(), 1);

  auto build_result = remap.remap_build_keys(build_table);
  auto host_ids     = to_host<int32_t>(*build_result);

  // All IDs should be unique and non-negative
  std::set<int32_t> unique_ids(host_ids.begin(), host_ids.end());
  EXPECT_EQ(unique_ids.size(), 5u);
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }
}

TEST_F(KeyRemappingTest, LargeTable)
{
  // Create a larger table to test with 100 distinct values, each appearing 100 times
  std::vector<int32_t> data(10000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<int32_t>(i % 100);
  }

  column_wrapper<int32_t> build_col(data.begin(), data.end());
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  EXPECT_EQ(remap.get_distinct_count(), 100);
  EXPECT_EQ(remap.get_max_duplicate_count(), 100);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 100);

  auto host_ids = to_host<int32_t>(*build_result);

  // Verify all rows with the same key have the same ID
  // Group by key value and check IDs
  std::map<int32_t, int32_t> key_to_id;
  for (size_t i = 0; i < data.size(); ++i) {
    auto key = data[i];
    auto id  = host_ids[i];
    if (key_to_id.count(key) == 0) {
      key_to_id[key] = id;
    } else {
      EXPECT_EQ(key_to_id[key], id) << "Key " << key << " has inconsistent IDs";
    }
  }

  // Probe with values 0-99 (all exist) and 100-104 (don't exist)
  std::vector<int32_t> probe_data;
  for (int i = 0; i < 105; ++i) {
    probe_data.push_back(i);
  }
  column_wrapper<int32_t> probe_col(probe_data.begin(), probe_data.end());
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  // Keys 0-99 should match build
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(host_probe_ids[i], key_to_id[i]) << "Probe key " << i << " mismatch";
  }
  // Keys 100-104 should be NOT_FOUND
  for (int i = 100; i < 105; ++i) {
    EXPECT_EQ(host_probe_ids[i], cudf::KEY_REMAP_NOT_FOUND)
      << "Probe key " << i << " should be NOT_FOUND";
  }
}

TEST_F(KeyRemappingTest, StructKeys)
{
  // Test with struct column keys
  column_wrapper<int32_t> child1{1, 1, 2, 1};
  strcol_wrapper child2{"a", "b", "a", "a"};
  auto struct_col  = cudf::test::structs_column_wrapper{{child1, child2}};
  auto build_table = cudf::table_view{{struct_col}};

  cudf::key_remapping remap{build_table};

  // Distinct structs: {1,"a"}, {1,"b"}, {2,"a"} = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  auto host_ids = to_host<int32_t>(*build_result);
  // Rows 0 and 3 have same struct {1,"a"}
  EXPECT_EQ(host_ids[0], host_ids[3]);
  // All different struct values have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);
}

TEST_F(KeyRemappingTest, FloatKeys)
{
  // Test with float keys including duplicates
  column_wrapper<float> build_col{1.5f, 2.5f, 3.5f, 2.5f, 1.5f};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  // Distinct: 1.5, 2.5, 3.5 = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  auto host_ids = to_host<int32_t>(*build_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[0], host_ids[4]);  // Both 1.5
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.5
  // Different keys should have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);

  // Probe with matching and non-matching keys
  column_wrapper<float> probe_col{3.5f, 1.5f, 9.9f, 2.5f};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  EXPECT_EQ(host_probe_ids[0], host_ids[2]);                // 3.5 matches
  EXPECT_EQ(host_probe_ids[1], host_ids[0]);                // 1.5 matches
  EXPECT_EQ(host_probe_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // 9.9 not found
  EXPECT_EQ(host_probe_ids[3], host_ids[1]);                // 2.5 matches

  verify_probe_matches_build(build_table, *build_result, probe_table, *probe_result);
}

TEST_F(KeyRemappingTest, DoubleKeys)
{
  // Test with double keys including duplicates
  column_wrapper<double> build_col{1.123456789, 2.987654321, 3.141592653, 2.987654321};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  // Distinct: 3 values (the second 2.987654321 is a duplicate)
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  auto host_ids = to_host<int32_t>(*build_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.987654321
  // Different keys should have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);

  // Probe with matching and non-matching keys
  column_wrapper<double> probe_col{3.141592653, 1.123456789, 99.99};
  auto probe_table = cudf::table_view{{probe_col}};

  auto probe_result   = remap.remap_probe_keys(probe_table);
  auto host_probe_ids = to_host<int32_t>(*probe_result);

  EXPECT_EQ(host_probe_ids[0], host_ids[2]);                // pi matches
  EXPECT_EQ(host_probe_ids[1], host_ids[0]);                // 1.123... matches
  EXPECT_EQ(host_probe_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // 99.99 not found

  verify_probe_matches_build(build_table, *build_result, probe_table, *probe_result);
}

TEST_F(KeyRemappingTest, FloatWithNulls)
{
  // Test float keys with null values
  column_wrapper<float> build_col{{1.5f, 2.5f, 0.0f, 2.5f}, {true, true, false, true}};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table, cudf::null_equality::EQUAL};

  // Distinct: 1.5, 2.5, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto build_result = remap.remap_build_keys(build_table);
  verify_remapping_contract(build_table, *build_result, 3);

  auto host_ids = to_host<int32_t>(*build_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.5
  // All IDs should be non-negative (nulls treated as equal)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Test with UNEQUAL null semantics
  cudf::key_remapping remap_unequal{build_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1.5, 2.5 = 2 (null skipped)
  EXPECT_EQ(remap_unequal.get_distinct_count(), 2);

  auto build_result_unequal = remap_unequal.remap_build_keys(build_table);
  auto host_ids_unequal     = to_host<int32_t>(*build_result_unequal);

  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids_unequal[2], cudf::KEY_REMAP_BUILD_NULL);
}

TEST_F(KeyRemappingTest, DoubleWithNulls)
{
  // Test double keys with null values
  column_wrapper<double> build_col{{1.0, 2.0, 0.0, 2.0}, {true, true, false, true}};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table, cudf::null_equality::EQUAL};

  // Distinct: 1.0, 2.0, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto build_result = remap.remap_build_keys(build_table);
  auto host_ids     = to_host<int32_t>(*build_result);

  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.0
  // All IDs should be non-negative (nulls treated as equal)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Test with UNEQUAL null semantics
  cudf::key_remapping remap_unequal{build_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1.0, 2.0 = 2 (null skipped)
  EXPECT_EQ(remap_unequal.get_distinct_count(), 2);

  auto build_result_unequal = remap_unequal.remap_build_keys(build_table);
  auto host_ids_unequal     = to_host<int32_t>(*build_result_unequal);

  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids_unequal[2], cudf::KEY_REMAP_BUILD_NULL);
}

// Schema validation tests: probe table must match build table schema

TEST_F(KeyRemappingTest, ProbeSchemaMismatchColumnCount)
{
  // Build with 2 columns
  column_wrapper<int32_t> build_col1{1, 2, 3};
  column_wrapper<int32_t> build_col2{4, 5, 6};
  auto build_table = cudf::table_view{{build_col1, build_col2}};

  cudf::key_remapping remap{build_table};

  // Probe with 1 column - should throw
  column_wrapper<int32_t> probe_col{1, 2};
  auto probe_table = cudf::table_view{{probe_col}};

  EXPECT_THROW((void)remap.remap_probe_keys(probe_table), std::invalid_argument);
  EXPECT_THROW((void)remap.remap_build_keys(probe_table), std::invalid_argument);
}

TEST_F(KeyRemappingTest, ProbeSchemaMismatchColumnType)
{
  // Build with INT32
  column_wrapper<int32_t> build_col{1, 2, 3};
  auto build_table = cudf::table_view{{build_col}};

  cudf::key_remapping remap{build_table};

  // Probe with INT64 - should throw due to type mismatch
  column_wrapper<int64_t> probe_col{1, 2, 3};
  auto probe_table = cudf::table_view{{probe_col}};

  EXPECT_THROW((void)remap.remap_probe_keys(probe_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_build_keys(probe_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, ProbeSchemaMismatchNestedVsPrimitive)
{
  // Build with struct column
  column_wrapper<int32_t> child1{1, 2, 3};
  strcol_wrapper child2{"a", "b", "c"};
  auto struct_col  = cudf::test::structs_column_wrapper{{child1, child2}};
  auto build_table = cudf::table_view{{struct_col}};

  cudf::key_remapping remap{build_table};

  // Probe with primitive column - should throw due to type mismatch
  column_wrapper<int32_t> probe_col{1, 2, 3};
  auto probe_table = cudf::table_view{{probe_col}};

  EXPECT_THROW((void)remap.remap_probe_keys(probe_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_build_keys(probe_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, ProbeSchemaMismatchStructFields)
{
  // Build with struct{INT32, STRING}
  column_wrapper<int32_t> build_child1{1, 2, 3};
  strcol_wrapper build_child2{"a", "b", "c"};
  auto build_struct = cudf::test::structs_column_wrapper{{build_child1, build_child2}};
  auto build_table  = cudf::table_view{{build_struct}};

  cudf::key_remapping remap{build_table};

  // Probe with struct{INT32, INT32} - different field types, should throw
  column_wrapper<int32_t> probe_child1{1, 2, 3};
  column_wrapper<int32_t> probe_child2{4, 5, 6};
  auto probe_struct = cudf::test::structs_column_wrapper{{probe_child1, probe_child2}};
  auto probe_table  = cudf::table_view{{probe_struct}};

  EXPECT_THROW((void)remap.remap_probe_keys(probe_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_build_keys(probe_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, EmptyProbeSchemaMismatchColumnCount)
{
  // Build with 2 columns
  column_wrapper<int32_t> build_col1{1, 2, 3};
  column_wrapper<int32_t> build_col2{4, 5, 6};
  auto build_table = cudf::table_view{{build_col1, build_col2}};

  cudf::key_remapping remap{build_table};

  // Empty probe with 1 column - should still throw due to column count mismatch
  column_wrapper<int32_t> probe_col{};
  auto probe_table = cudf::table_view{{probe_col}};

  EXPECT_THROW((void)remap.remap_probe_keys(probe_table), std::invalid_argument);
  EXPECT_THROW((void)remap.remap_build_keys(probe_table), std::invalid_argument);
}

// Tests for optional metrics computation

TEST_F(KeyRemappingTest, MetricsEnabled)
{
  column_wrapper<int32_t> build_col{1, 2, 2, 3, 3, 3};
  auto build_table = cudf::table_view{{build_col}};

  // Default: metrics enabled
  cudf::key_remapping remap{build_table};

  EXPECT_TRUE(remap.has_metrics());
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 3);
}

TEST_F(KeyRemappingTest, MetricsDisabled)
{
  column_wrapper<int32_t> build_col{1, 2, 2, 3, 3, 3};
  auto build_table = cudf::table_view{{build_col}};

  // Explicitly disable metrics
  cudf::key_remapping remap{build_table, cudf::null_equality::EQUAL, false};

  EXPECT_FALSE(remap.has_metrics());
  EXPECT_THROW((void)remap.get_distinct_count(), cudf::logic_error);
  EXPECT_THROW((void)remap.get_max_duplicate_count(), cudf::logic_error);
}

TEST_F(KeyRemappingTest, MetricsDisabledRemapStillWorks)
{
  column_wrapper<int32_t> build_col{10, 20, 20, 30};
  auto build_table = cudf::table_view{{build_col}};

  // Disable metrics but remapping should still work
  cudf::key_remapping remap{build_table, cudf::null_equality::EQUAL, false};

  // Remap build keys
  auto build_result = remap.remap_build_keys(build_table);
  auto build_ids    = to_host<int32_t>(build_result->view());

  // Equal keys should have equal IDs
  EXPECT_EQ(build_ids[1], build_ids[2]);  // Both 20s
  EXPECT_NE(build_ids[0], build_ids[1]);  // 10 vs 20
  EXPECT_NE(build_ids[1], build_ids[3]);  // 20 vs 30

  // Remap probe keys
  column_wrapper<int32_t> probe_col{20, 40, 10};
  auto probe_table  = cudf::table_view{{probe_col}};
  auto probe_result = remap.remap_probe_keys(probe_table);
  auto probe_ids    = to_host<int32_t>(probe_result->view());

  EXPECT_EQ(probe_ids[0], build_ids[1]);                  // 20 matches
  EXPECT_EQ(probe_ids[1], cudf::KEY_REMAP_NOT_FOUND);     // 40 not found
  EXPECT_EQ(probe_ids[2], build_ids[0]);                  // 10 matches
}
