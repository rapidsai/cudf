/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/filter.hpp>
#include <cudf/jit/runtime_support.hpp>

namespace filters {

struct FilterTest : public cudf::test::BaseFixture {
 protected:
  void SetUp() override
  {
    if (!cudf::is_runtime_jit_supported()) {
      GTEST_SKIP() << "Skipping tests that require runtime JIT support";
    }
  }
};

// [ ] type support tests

TEST_F(FilterTest, FilterEven)
{
  auto a           = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::string cuda = R"***(
    __device__ void is_even(bool* out, int32_t a) {
      *out = (a % 2 == 0);
    }
    )***";

  auto result   = cudf::filter({a}, {a}, cuda, false, std::nullopt);
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{2, 4, 6, 8, 10};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result[0]->view());
}

TEST_F(FilterTest, MultiFilter)
{
  auto countries = cudf::test::strings_column_wrapper{
    "USA", "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany", "Italy", "Spain"};
  auto average_tmp      = cudf::test::fixed_width_column_wrapper<float>{0, 0, 1, 1, 1, 1, 1, 1, 1};
  auto average_humidity = cudf::test::fixed_width_column_wrapper<float>{0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto timezones        = cudf::test::strings_column_wrapper{
    "EST", "EST", "CST", "BRT", "ART", "CET", "CET", "CET", "CET"};

  std::string cuda = R"***(
 __device__ void filter(bool* out, float tmp, float hum, cudf::string_view tz, float min_tmp, float max_tmp, float min_hum, float max_hum, cudf::string_view tz1, cudf::string_view tz2) {
    *out = (tmp >= min_tmp && tmp <= max_tmp) &&
           (hum >= min_hum && hum <= max_hum) &&
           (tz == tz1 || tz == tz2);
    }
  )***";

  auto min_tmp   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_tmp   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto min_hum   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_hum   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto timezone1 = cudf::test::strings_column_wrapper{"CET"};
  auto timezone2 = cudf::test::strings_column_wrapper{"EST"};

  auto result = cudf::filter({countries, timezones},
                             {average_tmp,
                              average_humidity,
                              timezones,
                              min_tmp,
                              max_tmp,
                              min_hum,
                              max_hum,
                              timezone1,
                              timezone2},
                             cuda,
                             false,
                             std::nullopt);

  auto expected_countries =
    cudf::test::strings_column_wrapper{"France", "Germany", "Italy", "Spain"};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_countries, result[0]->view());

  auto expected_timezones = cudf::test::strings_column_wrapper{"CET", "CET", "CET", "CET"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_timezones, result[1]->view());
}

}  // namespace filters
