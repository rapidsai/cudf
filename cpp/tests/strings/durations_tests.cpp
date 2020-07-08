/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/durations.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

#define CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, rhs) \
  {                                              \
    SCOPED_TRACE(" <--  line of failure\n");     \
    cudf::test::expect_columns_equal(lhs, rhs);  \
  }

struct StringsDurationsTest : public cudf::test::BaseFixture {
};

TEST_F(StringsDurationsTest, FromToDurations)
{
  using T = cudf::duration_s;
  std::vector<cudf::duration_s> h_durations{
    T{131246625}, T{1563399277}, T{0}, T{1553085296}, T{1582934400}, T{-1545730073}, T{-86399}};
  std::vector<const char*> h_expected{"1519 days 01:23:45",
                                      "18094 days 21:34:37",
                                      nullptr,
                                      "17975 days 12:34:56",
                                      "18321 days 00:00:00",
                                      "-17891 days +14:32:07",
                                      "-1 days +00:00:01"};

  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations(
    h_durations.begin(),
    h_durations.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_durations(durations);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  //
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%d days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, ISOFormat)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{1530705600L}, T{1582934461L}, T{1451430122L}, T{1318302183L}, T{-6105994200L}};
  auto results = cudf::strings::from_durations(durations, "P%dDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected{
    "P17716DT12H0M0S", "P18321DT0H1M1S", "P16798DT23H2M2S", "P15258DT3H3M3S", "P-70672DT18H30M0S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  //
  auto new_durations = cudf::strings::to_durations(
    cudf::strings_column_view(expected), cudf::data_type(cudf::type_to_id<T>()), "P%dDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, ISOFormatDaysOnly)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{17716L}, T{18321L}, T{16798L}, T{15258L}, T{-70672L}};
  auto results1 = cudf::strings::from_durations(durations, "P%dDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected1{
    "P17716DT0H0M0S", "P18321DT0H0M0S", "P16798DT0H0M0S", "P15258DT0H0M0S", "P-70672DT0H0M0S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results1, expected1);

  auto results2 = cudf::strings::from_durations(durations, "P%dD");
  cudf::test::strings_column_wrapper expected2{
    "P17716D", "P18321D", "P16798D", "P15258D", "P-70672D"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results2, expected2);

  //
  auto new_durations1 = cudf::strings::to_durations(
    cudf::strings_column_view(expected1), cudf::data_type(cudf::type_to_id<T>()), "P%dDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations1, durations);
  auto new_durations2 = cudf::strings::to_durations(
    cudf::strings_column_view(expected2), cudf::data_type(cudf::type_to_id<T>()), "P%dD");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations2, durations);
}

TEST_F(StringsDurationsTest, ISOFormatSubseconds)
{
  using T = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<T> durations{T{0L},
                                                      T{7000000000L},
                                                      T{11L},
                                                      T{10L},
                                                      T{17716L},
                                                      T{18321L},
                                                      T{16798L},
                                                      T{15258L},
                                                      T{15258000L},
                                                      T{-70672L}};
  // fully isoformat compliant.
  auto results = cudf::strings::from_durations(durations, "P%dDT%HH%MM%S%fS");
  cudf::test::strings_column_wrapper expected{"P0DT0H0M0S",
                                              "P0DT0H0M7S",
                                              "P0DT0H0M0.000000011S",
                                              "P0DT0H0M0.00000001S",
                                              "P0DT0H0M0.000017716S",
                                              "P0DT0H0M0.000018321S",
                                              "P0DT0H0M0.000016798S",
                                              "P0DT0H0M0.000015258S",
                                              "P0DT0H0M0.015258S",
                                              "P-1DT23H59M59.999929328S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  //
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "P%dDT%HH%MM%S%fS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationSeconds)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},          // 0 days 00:00:00
    T{1L},          // 0 days 00:00:01
    T{118800L},     // 1 days 09:00:00
    T{31568404L},   // 365 days 09:00:04
    T{-118800L},    // -2 days +15:00:00
    T{-31568404L},  // -366 days +14:59:56
  };
  auto results = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "0 days 00:00:01",
                                              "1 days 09:00:00",
                                              "365 days 09:00:04",
                                              "-2 days +15:00:00",
                                              "-366 days +14:59:56"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%d days %+%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationDays)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},            // 0 days 00:00:00
    T{1L},            // 1 days 00:00:00
    T{-1L},           // -1 days +00:00:00
    T{800L},          // 800 days 00:00:00
    T{-800L},         // -800 days +00:00:00
    T{2147483647L},   //  2147483647 days 00:00:00
    T{-2147483648L},  // -2147483648 days +00:00:00
  };
  auto results = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "1 days 00:00:00",
                                              "-1 days +00:00:00",
                                              "800 days 00:00:00",
                                              "-800 days +00:00:00",
                                              "2147483647 days 00:00:00",
                                              "-2147483648 days +00:00:00"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%d days %+%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationMilliseconds)
{
  using ms = cudf::duration_ms;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> durations_ms{ms{-60000},
                                                                         ms{1530705600123},
                                                                         ms{1582934461007},
                                                                         ms{1451430122420},
                                                                         ms{1451430122400},
                                                                         ms{1451430122000},
                                                                         ms{1318302183999},
                                                                         ms{-6106017600047}};
  auto results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S%3f");
  cudf::test::strings_column_wrapper expected_ms_3f{"-1 days +23:59:00.000",
                                                    "17716 days 12:00:00.123",
                                                    "18321 days 00:01:01.007",
                                                    "16798 days 23:02:02.420",
                                                    "16798 days 23:02:02.400",
                                                    "16798 days 23:02:02.000",
                                                    "15258 days 03:03:03.999",
                                                    "-70672 days +11:59:59.953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms_3f);

  results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S%6f");
  cudf::test::strings_column_wrapper expected_ms_6f{"-1 days +23:59:00.000000",
                                                    "17716 days 12:00:00.123000",
                                                    "18321 days 00:01:01.007000",
                                                    "16798 days 23:02:02.420000",
                                                    "16798 days 23:02:02.400000",
                                                    "16798 days 23:02:02.000000",
                                                    "15258 days 03:03:03.999000",
                                                    "-70672 days +11:59:59.953000"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms_6f);

  results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S%f");
  cudf::test::strings_column_wrapper expected_ms{"-1 days +23:59:00",
                                                 "17716 days 12:00:00.123",
                                                 "18321 days 00:01:01.007",
                                                 "16798 days 23:02:02.42",
                                                 "16798 days 23:02:02.4",
                                                 "16798 days 23:02:02",
                                                 "15258 days 03:03:03.999",
                                                 "-70672 days +11:59:59.953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);

  //
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms_3f),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%d days %+%H:%M:%S%3f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms_6f),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%d days %+%H:%M:%S%6f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%d days %+%H:%M:%S%f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
}

TEST_F(StringsDurationsTest, DurationMicroseconds)
{
  using us = cudf::duration_us;
  cudf::test::fixed_width_column_wrapper<cudf::duration_us> durations_us{us{-60000},
                                                                         us{1530705600123},
                                                                         us{1582934461007},
                                                                         us{1451430122420},
                                                                         us{1451430122400},
                                                                         us{1451430122000},
                                                                         us{1318302183999},
                                                                         us{-6106017600047}};
  auto results = cudf::strings::from_durations(durations_us, "%d days %+%H:%M:%S%3f");
  cudf::test::strings_column_wrapper expected_us_3f{"-1 days +23:59:59.940",
                                                    "17 days 17:11:45.600",
                                                    "18 days 07:42:14.461",
                                                    "16 days 19:10:30.122",
                                                    "16 days 19:10:30.122",
                                                    "16 days 19:10:30.122",
                                                    "15 days 06:11:42.183",
                                                    "-71 days +07:53:02.399"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us_3f);

  results = cudf::strings::from_durations(durations_us, "%d days %+%H:%M:%S%6f");
  cudf::test::strings_column_wrapper expected_us_6f{"-1 days +23:59:59.940000",
                                                    "17 days 17:11:45.600123",
                                                    "18 days 07:42:14.461007",
                                                    "16 days 19:10:30.122420",
                                                    "16 days 19:10:30.122400",
                                                    "16 days 19:10:30.122000",
                                                    "15 days 06:11:42.183999",
                                                    "-71 days +07:53:02.399953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us_6f);

  results = cudf::strings::from_durations(durations_us, "%d days %+%H:%M:%S%f");
  cudf::test::strings_column_wrapper expected_us{"-1 days +23:59:59.94",
                                                 "17 days 17:11:45.600123",
                                                 "18 days 07:42:14.461007",
                                                 "16 days 19:10:30.12242",
                                                 "16 days 19:10:30.1224",
                                                 "16 days 19:10:30.122",
                                                 "15 days 06:11:42.183999",
                                                 "-71 days +07:53:02.399953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us);

  //
  cudf::test::fixed_width_column_wrapper<cudf::duration_us> durations_us_3f{us{-60000},
                                                                            us{1530705600000},
                                                                            us{1582934461000},
                                                                            us{1451430122000},
                                                                            us{1451430122000},
                                                                            us{1451430122000},
                                                                            us{1318302183000},
                                                                            us{-6106017601000}};
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us_3f),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%d days %+%H:%M:%S%3f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us_3f);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us_6f),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%d days %+%H:%M:%S%6f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%d days %+%H:%M:%S%f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us);
}

TEST_F(StringsDurationsTest, DurationNanoseconds)
{
  using ns = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_ns{ns{1530705600123456789},
                                                                         ns{1582934461007008009},
                                                                         ns{1451430122421310209},
                                                                         ns{1318302183999777550},
                                                                         ns{-6106017600047047047}};
  auto results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S%9f");
  cudf::test::strings_column_wrapper expected_ns_9f{"17716 days 12:00:00.123456789",
                                                    "18321 days 00:01:01.007008009",
                                                    "16798 days 23:02:02.421310209",
                                                    "15258 days 03:03:03.999777550",
                                                    "-70672 days +11:59:59.952952953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_9f);

  results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S%6f");
  cudf::test::strings_column_wrapper expected_ns_6f{"17716 days 12:00:00.123456",
                                                    "18321 days 00:01:01.007008",
                                                    "16798 days 23:02:02.421310",
                                                    "15258 days 03:03:03.999777",
                                                    "-70672 days +11:59:59.952952"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_6f);

  results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S%f");
  cudf::test::strings_column_wrapper expected_ns{"17716 days 12:00:00.123456789",
                                                 "18321 days 00:01:01.007008009",
                                                 "16798 days 23:02:02.421310209",
                                                 "15258 days 03:03:03.99977755",
                                                 "-70672 days +11:59:59.952952953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);

  //
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns_9f),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%d days %+%H:%M:%S%9f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns);
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_ns_6f{
    ns{1530705600123456000},
    ns{1582934461007008000},
    ns{1451430122421310000},
    ns{1318302183999777000},
    ns{-6106017600047048000}};
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns_6f),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%d days %+%H:%M:%S%6f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns_6f);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%d days %+%H:%M:%S%f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns);
}

TEST_F(StringsDurationsTest, PandasMicroseconds)
{
  using T = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<T> durations{T{0L},
                                                      T{7000000000L},
                                                      T{11L},
                                                      T{10L},
                                                      T{17716L},
                                                      T{18321L},
                                                      T{16798L},
                                                      T{15258L},
                                                      T{15258000L},
                                                      T{-70672L}};
  // TODO: pandas does not have "0 days ".
  auto results = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S%u");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "0 days 00:00:07",
                                              "0 days 00:00:00.000000",
                                              "0 days 00:00:00.000000",
                                              "0 days 00:00:00.000017",
                                              "0 days 00:00:00.000018",
                                              "0 days 00:00:00.000016",
                                              "0 days 00:00:00.000015",
                                              "0 days 00:00:00.015258",
                                              "-1 days +23:59:59.999929"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDurationsTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::type_id::DURATION_SECONDS}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::from_durations(zero_size_column);
  cudf::test::expect_strings_empty(results->view());

  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  results = cudf::strings::to_durations(cudf::strings_column_view(zero_size_strings_column),
                                        cudf::data_type{cudf::type_id::DURATION_SECONDS},
                                        "%S");
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsDurationsTest, Errors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view view(strings);
  EXPECT_THROW(cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::INT64}, "%d"),
               cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, ""),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, "%2H"),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, "%g"),
    cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int64_t> invalid_durations{1530705600};
  EXPECT_THROW(cudf::strings::from_durations(invalid_durations), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations{cudf::duration_s{1530705600}};
  EXPECT_THROW(cudf::strings::from_durations(durations, ""), cudf::logic_error);
}
