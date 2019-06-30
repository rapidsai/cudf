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

#include <cudf/cudf.h>
#include <nvstrings/NVStrings.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests/io/io_test_utils.hpp>
#include <tests/utilities/cudf_test_fixtures.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
   ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

MATCHER_P(FloatNearPointwise, tolerance, "Out of range")
{
    return (std::get<0>(arg)>std::get<1>(arg)-tolerance &&
            std::get<0>(arg)<std::get<1>(arg)+tolerance) ;
}

TEST(gdf_csv_test, DetectColumns)
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

TEST(gdf_csv_test, UseColumns)
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

TEST(gdf_csv_test, Numbers) {
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

TEST(gdf_csv_test, MortPerf)
{
    const std::string fname = "Performance_2000Q1.txt";
    if (checkFile(fname))
    {
        cudf::csv_read_arg args(cudf::source_info{fname});
        args.delimiter = '|';
        cudf::read_csv(args);
    }
}

TEST(gdf_csv_test, Strings)
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

TEST(gdf_csv_test, QuotedStrings)
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

TEST(gdf_csv_test, IgnoreQuotes)
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

TEST(gdf_csv_test, Booleans)
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

TEST(gdf_csv_test, Dates)
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

TEST(gdf_csv_test, FloatingPoint)
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

TEST(gdf_csv_test, Category)
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

TEST(gdf_csv_test, SkiprowsNrows)
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

TEST(gdf_csv_test, ByteRange)
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

TEST(gdf_csv_test, BlanksAndComments)
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

TEST(gdf_csv_test, Writer)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvWriteTest.csv";

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "true,1,1.0,one" << '\n';
    outfile << "false,2,2.25,two" << '\n';
    outfile << "false,3,3.50,three" << '\n';
    outfile << "true,4,4.75,four" << '\n';
    outfile << "false,5,5.0,five" << '\n';
    outfile.close();

    cudf::csv_read_arg rargs(cudf::source_info{fname});
    rargs.names = { "boolean", "integer", "float", "string" };
    rargs.dtype = { "bool", "int32", "float32", "str" };
    rargs.header = -1;
    const auto df = cudf::read_csv(rargs);

    const std::string ofname = temp_env->get_temp_dir()+"CsvWriteTestOut.csv";
    csv_write_arg wargs{};
    wargs.columns = &(*df.begin());  // columns from reader above
    wargs.filepath = ofname.c_str();
    wargs.num_cols = df.num_columns();
    wargs.delimiter = ',';
    wargs.line_terminator = "\n";
    wargs.include_header = true;

    EXPECT_EQ( write_csv(&wargs), GDF_SUCCESS );

    std::ifstream infile(ofname);
    std::string csv((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    std::string verify =
        "\"boolean\",\"integer\",\"float\",\"string\"\n"
        "true,1,1.0,\"one\"\n"
        "false,2,2.25,\"two\"\n"
        "false,3,3.5,\"three\"\n"
        "true,4,4.75,\"four\"\n"
        "false,5,5.0,\"five\"\n";
    EXPECT_STREQ( csv.c_str(), verify.c_str() );
}
