/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cudf/cudf.h>
#include <cudf/unary.hpp>
#include <nvstrings/NVStrings.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/io/io_test_utils.hpp>
#include <tests/utilities/cudf_test_fixtures.h>

#include <arrow/io/api.h>

TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

/**
 * @brief Base test fixture for CSV reader/writer tests
 **/
struct CsvTest : public GdfTest {};

/**
 * @brief Test fixture for source content parameterized tests
 **/
struct CsvValueParamTest : public CsvTest,
                           public testing::WithParamInterface<const char*> {};

MATCHER_P(FloatNearPointwise, tolerance, "Out of range")
{
    return (std::get<0>(arg)>std::get<1>(arg)-tolerance &&
            std::get<0>(arg)<std::get<1>(arg)+tolerance) ;
}

TEST_F(CsvTest, DetectColumns)
{
    const std::string fname	= temp_env->get_temp_dir()+"DetectColumnsTest.csv";

    // types are  { "int", "float64", "int" };
    std::ofstream outfile(fname, std::ofstream::out);
    outfile << " 20, 0.40, 100\n"\
               "-21,-0.41, 101\n"\
               " 22, 0.42, 102\n"\
               "-23,-0.43, 103\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A", "B", "C" };
        args.header = -1;
        args.use_cols_names = { "A", "C" };
        const auto df = cudf::read_csv(args);

        // cudf auto detect type code uses INT64
        ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
        ASSERT_EQ(df.get_column(1)->dtype, GDF_INT64);
        auto ACol = gdf_host_column<int64_t>(df.get_column(0));
        auto BCol = gdf_host_column<int64_t>(df.get_column(1));
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre<int64_t>(20, -21, 22, -23) );
        EXPECT_THAT( BCol.hostdata(), ::testing::ElementsAre<int64_t>(100, 101, 102, 103) );
    }
}

TEST_F(CsvTest, UseColumns)
{
    const std::string fname	= temp_env->get_temp_dir()+"UseColumnsTest.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << " 20, 0.40, 100\n"\
               "-21,-0.41, 101\n"\
               " 22, 0.42, 102\n"\
               "-23,-0.43, 103\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A", "B", "C" };
        args.dtype = { "int", "float64", "int" };
        args.header = -1;
        args.use_cols_names = { "A", "C" };
        const auto df = cudf::read_csv(args);

        ASSERT_EQ( df.get_column(0)->dtype, GDF_INT32 );
        ASSERT_EQ( df.get_column(1)->dtype, GDF_INT32 );
        auto ACol = gdf_host_column<int32_t>(df.get_column(0));
        auto BCol = gdf_host_column<int32_t>(df.get_column(1));
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre<int32_t>(20, -21, 22, -23) );
        EXPECT_THAT( BCol.hostdata(), ::testing::ElementsAre<int32_t>(100, 101, 102, 103) );
    }
}

TEST_F(CsvTest, Numbers) {
  const std::string fname = temp_env->get_temp_dir() + "CsvNumbersTest.csv";

  constexpr int num_rows = 4;
  auto int8_values = random_values<int8_t>(num_rows);
  auto int16_values = random_values<int16_t>(num_rows);
  auto int32_values = random_values<int32_t>(num_rows);
  auto int64_values = random_values<int64_t>(num_rows);
  auto float32_values = random_values<float>(num_rows);
  auto float64_values = random_values<double>(num_rows);

  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << ","
           << int16_values[i] << "," << int16_values[i] << ","
           << int32_values[i] << "," << int32_values[i] << ","
           << int64_values[i] << "," << int64_values[i] << ","
           << float32_values[i] << "," << float32_values[i] << ","
           << float64_values[i] << "," << float64_values[i] << "\n";
    }
    std::ofstream outfile(fname, std::ofstream::out);
    outfile << line.str();
    outfile.close();
    ASSERT_TRUE(checkFile(fname));
  }

  {
    cudf::csv_read_arg args(cudf::source_info{fname});
    args.dtype = {"int8",    "short",  "int16",  "int",
                  "int32",   "long",   "int64",  "float",
                  "float32", "double", "float64"};
    args.header = -1;
    const auto df = cudf::read_csv(args);

    EXPECT_THAT(gdf_host_column<int8_t>(df.get_column(0)).hostdata(),
                ::testing::ElementsAreArray(int8_values));
    EXPECT_THAT(gdf_host_column<int16_t>(df.get_column(2)).hostdata(),
                ::testing::ElementsAreArray(int16_values));
    EXPECT_THAT(gdf_host_column<int32_t>(df.get_column(4)).hostdata(),
                ::testing::ElementsAreArray(int32_values));
    EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(6)).hostdata(),
                ::testing::ElementsAreArray(int64_values));
    EXPECT_THAT(gdf_host_column<float>(df.get_column(8)).hostdata(),
                ::testing::Pointwise(FloatNearPointwise(1e-5), float32_values));
    EXPECT_THAT(gdf_host_column<double>(df.get_column(10)).hostdata(),
                ::testing::Pointwise(FloatNearPointwise(1e-5), float64_values));
  }
}

TEST_F(CsvTest, MortPerf)
{
    const std::string fname = "Performance_2000Q1.txt";
    if (checkFile(fname))
    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.delimiter = '|';
        cudf::read_csv(args);
    }
}

TEST_F(CsvTest, Strings)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvStringsTest.csv";
    std::vector<std::string> names{"line", "verse"};

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << "30,stu \"\"vwx\"\" yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = names;
        args.dtype = { "int32", "str" };
        args.quoting = cudf::csv_read_arg::quote_style::QUOTE_NONE;
        const auto df = cudf::read_csv(args);

        // No filtering of any columns
        EXPECT_EQ( df.num_columns(), static_cast<int>(names.size()) );

        checkStrColumn(df.get_column(1), {"abc def ghi", "\"jkl mno pqr\"", "stu \"\"vwx\"\" yz"});
    }
}

TEST_F(CsvTest, QuotedStrings)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvQuotedStringsTest.csv";
    std::vector<std::string> names{ "line", "verse" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = names;
        args.dtype = { "int32", "str" };
        args.quotechar = '`';
        const auto df = cudf::read_csv(args);

        // No filtering of any columns
        EXPECT_EQ( df.num_columns(), static_cast<int>(names.size()) );

        checkStrColumn(df.get_column(1), {"abc,\ndef, ghi", "jkl, `mno`, pqr", "stu `vwx` yz"});
    }
}

TEST_F(CsvTest, IgnoreQuotes)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvIgnoreQuotesTest.csv";
    std::vector<std::string> names{ "line", "verse" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << "20,\"jkl \"\"mno\"\" pqr\"" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = names;
        args.dtype = { "int32", "str" };
        args.quoting = cudf::csv_read_arg::quote_style::QUOTE_NONE;
        args.doublequote = false; // do not replace double quotechar with single
        const auto df = cudf::read_csv(args);

        // No filtering of any columns
        EXPECT_EQ( df.num_columns(), static_cast<int>(names.size()) );

        checkStrColumn(df.get_column(1), {"\"abcdef ghi\"", "\"jkl \"\"mno\"\" pqr\"", "stu \"vwx\" yz"});
    }
}

TEST_F(CsvTest, Booleans)
{
    const std::string fname = temp_env->get_temp_dir() + "CsvBooleansTest.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
              "true\nYes,5,foo,false\n";
    outfile.close();
    ASSERT_TRUE(checkFile(fname));

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = {"A", "B", "C", "D"};
        args.dtype = {"int32", "int32", "short", "bool"};
        args.true_values = {"yes", "Yes", "YES", "foo", "FOO"};
        args.false_values = {"no", "No", "NO", "Bar", "bar"};
        args.header = -1;
        const auto df = cudf::read_csv(args);

        // Booleans are the same (integer) data type, but valued at 0 or 1
        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_INT32 );
        ASSERT_EQ( df.get_column(2)->dtype, GDF_INT16 );
        ASSERT_EQ( df.get_column(3)->dtype, GDF_BOOL8 );

        auto firstCol = gdf_host_column<int32_t>(df.get_column(0));
        EXPECT_THAT(firstCol.hostdata(), ::testing::ElementsAre(1, 0, 0, 0, 1));
        auto thirdCol = gdf_host_column<int16_t>(df.get_column(2));
        EXPECT_THAT(thirdCol.hostdata(), ::testing::ElementsAre(0, 1, 1, 0, 1));
        auto fourthCol = gdf_host_column<cudf::bool8>(df.get_column(3));
        EXPECT_THAT(
            fourthCol.hostdata(),
            ::testing::ElementsAre(cudf::true_v, cudf::true_v, cudf::false_v,
                                  cudf::true_v, cudf::false_v));
    }
}

TEST_F(CsvTest, Dates)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvDatesTest.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "date" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_DATE64 );

        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400000, 1288483200000, 782611200000,
                       656208000000, 0, 798163200000, 774144000000,
                       1149679230400, 1126875750400, 2764800000) );
    }
}

TEST_F(CsvTest, Timestamps)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvTimestamps.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1288508400000\n988873200000\n782636400000\n656233200000\n28800000\n1462003323000\n1391279423000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "timestamp" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_ms );
        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(1288508400000, 988873200000, 782636400000,
                                 656233200000, 28800000, 1462003323000, 1391279423000) );

    }
}

TEST(gdf_csv_test, TimestampSeconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvTimestampSeconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1288508400\n988873200\n782636400\n656233200\n28800\n1462003323\n1391279423\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "timestamp[s]" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_s );
        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(1288508400, 988873200, 782636400,
                                 656233200, 28800, 1462003323, 1391279423) );

    }
}

TEST(gdf_csv_test, TimestampMilliseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvTimestampMilliseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1288508400000\n988873200000\n782636400000\n656233200000\n28800000\n1462003323000\n1391279423000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "timestamp[ms]" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_ms );
        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(1288508400000, 988873200000, 782636400000,
                                 656233200000, 28800000, 1462003323000, 1391279423000) );

    }
}

TEST(gdf_csv_test, TimestampMicroseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvTimestampMicroseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1288508400000000\n988873200000000\n782636400000000\n656233200000000\n28800000000\n1462003323000000\n1391279423000000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "timestamp[us]" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_us );
        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(1288508400000000, 988873200000000, 782636400000000,
                                 656233200000000, 28800000000, 1462003323000000, 1391279423000000) );

    }
}

TEST(gdf_csv_test, TimestampNanoseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvTimestampNanoseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1288508400000000000\n988873200000000000\n782636400000000000\n656233200000000000\n28800000000000\n1462003323000000000\n1391279423000000000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "timestamp[ns]" };
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_ns );
        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(1288508400000000000, 988873200000000000, 782636400000000000,
                                 656233200000000000, 28800000000000, 1462003323000000000, 1391279423000000000) );

    }
}

TEST(gdf_csv_test, DatesCastToTimestampSeconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CastToTimestampSeconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "date" };
        args.out_time_unit = TIME_UNIT_s;
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_s );

        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400, 1288483200, 782611200,
                       656208000, 0, 798163200, 774144000,
                       1149679230, 1126875750, 2764800) );
    }
}

TEST(gdf_csv_test, DatesCastToTimestampMilliseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CastToTimestampMilliseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "date" };
        args.out_time_unit = TIME_UNIT_ms;
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_ms );

        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400000, 1288483200000, 782611200000,
                       656208000000, 0, 798163200000, 774144000000,
                       1149679230400, 1126875750400, 2764800000) );
    }
}

TEST(gdf_csv_test, DatesCastToTimestampMicroseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CastToTimestampMicroseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "date" };
        args.out_time_unit = TIME_UNIT_us;
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_us );

        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400000000, 1288483200000000, 782611200000000,
                       656208000000000, 0, 798163200000000, 774144000000000,
                       1149679230400000, 1126875750400000, 2764800000000) );
    }
}

TEST(gdf_csv_test, DatesCastToTimestampNanoseconds)
{
    const std::string fname = temp_env->get_temp_dir()+"CastToTimestampNanoseconds.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "date" };
        args.out_time_unit = TIME_UNIT_ns;
        args.dayfirst = true;
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_TIMESTAMP );
        ASSERT_EQ( df.get_column(0)->dtype_info.time_unit, TIME_UNIT_ns );

        auto ACol = gdf_host_column<uint64_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400000000000, 1288483200000000000, 782611200000000000,
                       656208000000000000, 0, 798163200000000000, 774144000000000000,
                       1149679230400000000, 1126875750400000000, 2764800000000000) );
    }
}

TEST_F(CsvTest, FloatingPoint)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvFloatingPoint.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73.98007199999998;";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "float32" };
        args.lineterminator = ';';
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_FLOAT32 );

        auto ACol = gdf_host_column<float>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
            ::testing::Pointwise(FloatNearPointwise(1e-6),
                std::vector<float>{ 5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31, -73.98007199999998 }) );
    }
}

TEST_F(CsvTest, Category)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvCategory.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "HBM0676;KRC0842;ILM1441;EJV0094;";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "UserID" };
        args.dtype = { "category" };
        args.lineterminator = ';';
        args.header = -1;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_CATEGORY );

        auto ACol = gdf_host_column<int32_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(),
            ::testing::ElementsAre(2022314536, -189888986, 1512937027, 397836265) );
    }
}

TEST_F(CsvTest, SkiprowsNrows)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvSkiprowsNrows.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "int32" };
        args.header = 1;
        args.skiprows = 2;
        args.skipfooter = 0;
        args.nrows = 2;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(5, 6) );
    }
}

TEST_F(CsvTest, ByteRange)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvByteRange.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "int32" };
        args.header = -1;
        args.byte_range_offset = 11;
        args.byte_range_size = 15;
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(4000, 5000, 6000) );
    }
}

TEST_F(CsvTest, BlanksAndComments)
{
    const std::string fname = temp_env->get_temp_dir()+"BlanksAndComments.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.names = { "A" };
        args.dtype = { "int32" };
        args.header = -1;
        args.comment = '#';
        const auto df = cudf::read_csv(args);

        EXPECT_EQ( df.num_columns(), static_cast<int>(args.names.size()) );
        ASSERT_EQ( df.get_column(0)->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(df.get_column(0));
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(1, 3, 4, 5, 8, 9) );
    }
}

TEST_P(CsvValueParamTest, EmptyFileSource) {
  const std::string fname = temp_env->get_temp_dir() + "EmptyFileSource.csv";

  std::ofstream outfile{fname, std::ofstream::out};
  outfile << GetParam();
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  cudf::csv_read_arg args(cudf::source_info{fname});
  const auto df = cudf::read_csv(args);
  EXPECT_EQ(0, df.num_columns());
}
INSTANTIATE_TEST_CASE_P(CsvReader, CsvValueParamTest,
                        testing::Values("", "\n"));

TEST_F(CsvTest, ArrowFileSource) {
  const std::string fname = temp_env->get_temp_dir() + "ArrowFileSource.csv";

  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "A\n9\n8\n7\n6\n5\n4\n3\n2\n";
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(fname, &infile).ok());

  cudf::csv_read_arg args(cudf::source_info{infile});
  args.dtype = {"int8"};
  const auto df = cudf::read_csv(args);

  EXPECT_EQ(df.num_columns(), static_cast<gdf_size_type>(args.dtype.size()));
  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT8);

  const auto col = gdf_host_column<int8_t>(df.get_column(0));
  EXPECT_THAT(col.hostdata(), ::testing::ElementsAre(9, 8, 7, 6, 5, 4, 3, 2));
}

TEST_F(CsvTest, Writer)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvWriteTest.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "boolean,integer,long,short,byte,float,double,string,datetime" << '\n';
    outfile << "true,111,1111,11,1,1.0,12.0,one,01/01/2001" << '\n';
    outfile << "false,222,2222,22,2,2.25,24.0,two,02/02/2002" << '\n';
    outfile << "false,333,3333,33,3,3.50,32.0,three,03/03/2003" << '\n';
    outfile << "true,444,4444,44,4,4.75,48.0,four,04/04/2004" << '\n';
    outfile << "false,555,5555,55,5,5.0,56.0,five,05/05/2005" << '\n';
    outfile << "false,666,6666,66,6,6.125,64.0,six,06/06/2006" << '\n';
    outfile << "false,777,7777,77,7,7.25,72.0,seven,07/07/2007" << '\n';
    outfile << "true,888,8888,88,8,8.5,80.0,eight,08/08/2008" << '\n';
    outfile << "true,999,9999,99,9,9.75,92.0,nine,09/09/2009 09:09:09.009" << '\n';
    outfile << "false,1111,11111,111,10,10.5,108.0,ten,10/10/2010 10:10:10.010" << '\n';
    outfile.close();

    cudf::csv_read_arg rargs(cudf::source_info{fname});
    rargs.names = { "boolean", "integer", "long", "short", "byte", "float", "double", "string", "datetime" };
    rargs.dtype = { "bool", "int32", "int64", "int16", "int8", "float32", "float64", "str", "date" };
    rargs.header = 0;
    const auto df = cudf::read_csv(rargs);

    const std::string ofname = temp_env->get_temp_dir()+"CsvWriteTestOut.csv";
    csv_write_arg wargs{};
    wargs.columns = &(*df.begin());  // columns from reader above
    wargs.filepath = ofname.c_str();
    wargs.num_cols = df.num_columns();
    wargs.delimiter = ',';
    wargs.na_rep = "";
    wargs.line_terminator = "\n";
    wargs.include_header = true;
    wargs.rows_per_chunk = 8;

    EXPECT_EQ( write_csv(&wargs), GDF_SUCCESS );

    std::ifstream infile(ofname);
    std::string csv((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    std::string verify =
        "\"boolean\",\"integer\",\"long\",\"short\",\"byte\",\"float\",\"double\",\"string\",\"datetime\"\n"
        "true,111,1111,11,1,1.0,12.0,\"one\",\"2001-01-01T00:00:00Z\"\n"
        "false,222,2222,22,2,2.25,24.0,\"two\",\"2002-02-02T00:00:00Z\"\n"
        "false,333,3333,33,3,3.5,32.0,\"three\",\"2003-03-03T00:00:00Z\"\n"
        "true,444,4444,44,4,4.75,48.0,\"four\",\"2004-04-04T00:00:00Z\"\n"
        "false,555,5555,55,5,5.0,56.0,\"five\",\"2005-05-05T00:00:00Z\"\n"
        "false,666,6666,66,6,6.125,64.0,\"six\",\"2006-06-06T00:00:00Z\"\n"
        "false,777,7777,77,7,7.25,72.0,\"seven\",\"2007-07-07T00:00:00Z\"\n"
        "true,888,8888,88,8,8.5,80.0,\"eight\",\"2008-08-08T00:00:00Z\"\n"
        "true,999,9999,99,9,9.75,92.0,\"nine\",\"2009-09-09T09:09:09Z\"\n"
        "false,1111,11111,111,10,10.5,108.0,\"ten\",\"2010-10-10T10:10:10Z\"\n";
    EXPECT_STREQ( csv.c_str(), verify.c_str() );
}
