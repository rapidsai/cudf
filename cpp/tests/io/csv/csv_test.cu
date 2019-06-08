/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <tests/utilities/cudf_test_fixtures.h>

#include <nvstrings/NVStrings.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <sys/stat.h>

TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
   ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

MATCHER_P(FloatNearPointwise, tolerance, "Out of range")
{
    return (std::get<0>(arg)>std::get<1>(arg)-tolerance &&
            std::get<0>(arg)<std::get<1>(arg)+tolerance) ;
}

namespace {

bool checkFile(const std::string fname)
{
    struct stat st;
    return (stat(fname.c_str(), &st) ? 0 : 1);
}

template <typename T>
auto random_values(size_t size) {
  std::vector<T> values(size);

  using uniform_distribution =
      typename std::conditional<std::is_integral<T>::value,
                                std::uniform_int_distribution<T>,
                                std::uniform_real_distribution<T>>::type;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return dist(engine); });

  return values;
}

}  // namespace

// DESCRIPTION: Simple test internal helper class to transfer cudf column data
// from device to host for test comparisons and debugging/development
template <typename T>
class gdf_host_column
{
public:
    gdf_host_column() = delete;
    explicit gdf_host_column(gdf_column* const col)
    {
        m_hostdata = std::vector<T>(col->size);
        cudaMemcpy(m_hostdata.data(), col->data, sizeof(T) * col->size, cudaMemcpyDeviceToHost);
    }

    auto hostdata() const -> const auto&
    {
        return m_hostdata;
    }
    void print() const
    {
        for (size_t i = 0; i < m_hostdata.size(); ++i)
        {
            std::cout.precision(17);
            std::cout << "[" << i << "]: value=" << m_hostdata[i] << "\n";
        }
    }

private:
    std::vector<T> m_hostdata;
};

TEST(gdf_csv_test, DetectColumns)
{
    const std::string fname	= temp_env->get_temp_dir()+"DetectColumnsTest.csv";
    const char* names[]	= { "A", "B", "C" };
    const char* use_cols[]	= { "A", "C" };

    // types are  { "int", "float64", "int" };
    std::ofstream outfile(fname, std::ofstream::out);
    outfile << " 20, 0.40, 100\n"\
               "-21,-0.41, 101\n"\
               " 22, 0.42, 102\n"\
               "-23,-0.43, 103\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form    = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.dtype = NULL;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.decimal = '.';
        args.skip_blank_lines = true;
        args.header = -1;
        args.nrows = -1;
        args.use_cols_char = use_cols;
        args.use_cols_char_len = 2;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        // cudf auto detect type code uses INT64
        ASSERT_EQ( args.data[0]->dtype, GDF_INT64 );
        ASSERT_EQ( args.data[1]->dtype, GDF_INT64 );
        auto ACol = gdf_host_column<int64_t>(args.data[0]);
        auto BCol = gdf_host_column<int64_t>(args.data[1]);
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre<int64_t>(20, -21, 22, -23) );
        EXPECT_THAT( BCol.hostdata(), ::testing::ElementsAre<int64_t>(100, 101, 102, 103) );
    }
}

TEST(gdf_csv_test, UseColumns)
{
    const std::string fname	= temp_env->get_temp_dir()+"UseColumnsTest.csv";
    const char* names[]	= { "A", "B", "C" };
    const char* types[]	= { "int", "float64", "int" };
    const char* use_cols[]	= { "A", "C" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << " 20, 0.40, 100\n"\
               "-21,-0.41, 101\n"\
               " 22, 0.42, 102\n"\
               "-23,-0.43, 103\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form    = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = std::extent<decltype(types)>::value;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.decimal = '.';
        args.skip_blank_lines = true;
        args.header = -1;
        args.nrows = -1;
        args.use_cols_char = use_cols;
        args.use_cols_char_len = 2;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );
        ASSERT_EQ( args.data[1]->dtype, GDF_INT32 );
        auto ACol = gdf_host_column<int32_t>(args.data[0]);
        auto BCol = gdf_host_column<int32_t>(args.data[1]);
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre<int32_t>(20, -21, 22, -23) );
        EXPECT_THAT( BCol.hostdata(), ::testing::ElementsAre<int32_t>(100, 101, 102, 103) );
    }
}

TEST(gdf_csv_test, Numbers) {
  const std::string fname = temp_env->get_temp_dir() + "CsvNumbersTest.csv";
  const char* types[] = {"int8",    "short",  "int16",  "int",
                         "int32",   "long",   "int64",  "float",
                         "float32", "double", "float64"};

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
    csv_read_arg args{};
    args.input_data_form = gdf_csv_input_form::FILE_PATH;
    args.filepath_or_buffer = fname.c_str();
    args.num_dtype = std::extent<decltype(types)>::value;
    args.dtype = types;
    args.delimiter = ',';
    args.lineterminator = '\n';
    args.decimal = '.';
    args.skip_blank_lines = true;
    args.header = -1;
    args.nrows = -1;
    ASSERT_EQ(read_csv(&args), GDF_SUCCESS);

    EXPECT_THAT(gdf_host_column<int8_t>(args.data[0]).hostdata(),
                ::testing::ElementsAreArray(int8_values));
    EXPECT_THAT(gdf_host_column<int16_t>(args.data[2]).hostdata(),
                ::testing::ElementsAreArray(int16_values));
    EXPECT_THAT(gdf_host_column<int32_t>(args.data[4]).hostdata(),
                ::testing::ElementsAreArray(int32_values));
    EXPECT_THAT(gdf_host_column<int64_t>(args.data[6]).hostdata(),
                ::testing::ElementsAreArray(int64_values));
    EXPECT_THAT(gdf_host_column<float>(args.data[8]).hostdata(),
                ::testing::Pointwise(FloatNearPointwise(1e-5), float32_values));
    EXPECT_THAT(gdf_host_column<double>(args.data[10]).hostdata(),
                ::testing::Pointwise(FloatNearPointwise(1e-5), float64_values));
  }
}

TEST(gdf_csv_test, MortPerf)
{
    gdf_error error = GDF_SUCCESS;

    csv_read_arg	args{};
    const int num_cols = 31;

    args.num_names = num_cols;
    args.nrows = -1;

    const char ** dnames = new const char *[num_cols] {
        "loan_id",
        "monthly_reporting_period",
        "servicer",
        "interest_rate",
        "current_actual_upb",
        "loan_age",
        "remaining_months_to_legal_maturity",
        "adj_remaining_months_to_maturity",
        "maturity_date",
        "msa",
        "current_loan_delinquency_status",
        "mod_flag",
        "zero_balance_code",
        "zero_balance_effective_date",
        "last_paid_installment_date",
        "foreclosed_after",
        "disposition_date",
        "foreclosure_costs",
        "prop_preservation_and_repair_costs",
        "asset_recovery_costs",
        "misc_holding_expenses",
        "holding_taxes",
        "net_sale_proceeds",
        "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds",
        "other_foreclosure_proceeds",
        "non_interest_bearing_upb",
        "principal_forgiveness_upb",
        "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount",
        "servicing_activity_indicator"
    };
    args.names = dnames;

    args.num_dtype = num_cols;
    const char ** dtype = new const char *[num_cols] {
            "int64",
            "date",
            "category",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "date",
            "float64",
            "category",
            "category",
            "category",
            "date",
            "date",
            "date",
            "date",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "category",
            "float64",
            "category"
        };

        args.dtype = dtype;

        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = (char *)("Performance_2000Q1.txt");

    if (  checkFile(args.filepath_or_buffer))
    {
        args.delimiter 		= '|';
        args.lineterminator = '\n';
        args.delim_whitespace = 0;
        args.skipinitialspace = 0;
        args.skiprows 		= 0;
        args.skipfooter 	= 0;
        args.dayfirst 		= 0;
        args.mangle_dupe_cols=true;
        args.num_cols_out=0;

        args.use_cols_int       = NULL;
        args.use_cols_char      = NULL;
        args.use_cols_char_len  = 0;
        args.use_cols_int_len   = 0;


        args.names = NULL;
        args.dtype = NULL;


        error = read_csv(&args);
    }

    EXPECT_TRUE( error == GDF_SUCCESS );
}

TEST(gdf_csv_test, Strings)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvStringsTest.csv";
    const char* names[]	= { "line", "verse" };
    const char* types[]	= { "int32", "str" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << "30,stu \"\"vwx\"\" yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.skip_blank_lines = true;
        args.header = 0;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        // No filtering of any columns
        EXPECT_EQ( args.num_cols_out, args.num_names );

        // Check the parsed string column metadata
        ASSERT_EQ( args.data[1]->dtype, GDF_STRING );
        auto stringList = reinterpret_cast<NVStrings*>(args.data[1]->data);

        ASSERT_NE( stringList, nullptr );
        auto stringCount = stringList->size();
        ASSERT_EQ( stringCount, 3u );
        auto stringLengths = std::unique_ptr<int[]>{ new int[stringCount] };
        ASSERT_NE( stringList->byte_count(stringLengths.get(), false), 0u );

        // Check the actual strings themselves
        auto strings = std::unique_ptr<char*[]>{ new char*[stringCount] };
        for (size_t i = 0; i < stringCount; ++i) {
            ASSERT_GT( stringLengths[i], 0 );
            strings[i] = new char[stringLengths[i] + 1];
            strings[i][stringLengths[i]] = 0;
        }
        EXPECT_EQ( stringList->to_host(strings.get(), 0, stringCount), 0 );

        EXPECT_STREQ( strings[0], "abc def ghi" );
        EXPECT_STREQ( strings[1], "\"jkl mno pqr\"" );
        EXPECT_STREQ( strings[2], "stu \"\"vwx\"\" yz" );
        for (size_t i = 0; i < stringCount; ++i) {
            delete[] strings[i];
        }
    }
}

TEST(gdf_csv_test, QuotedStrings)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvQuotedStringsTest.csv";
    const char* names[]	= { "line", "verse" };
    const char* types[]	= { "int32", "str" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.quotechar = '`';
        args.quoting = QUOTE_ALL;     // enable quoting
        args.doublequote = true;      // replace double quotechar with single
        args.skip_blank_lines = true;
        args.header = 0;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        // No filtering of any columns
        EXPECT_EQ( args.num_cols_out, args.num_names );

        // Check the parsed string column metadata
        ASSERT_EQ( args.data[1]->dtype, GDF_STRING );
        auto stringList = reinterpret_cast<NVStrings*>(args.data[1]->data);

        ASSERT_NE( stringList, nullptr );
        auto stringCount = stringList->size();
        ASSERT_EQ( stringCount, 3u );
        auto stringLengths = std::unique_ptr<int[]>{ new int[stringCount] };
        ASSERT_NE( stringList->len(stringLengths.get(), false), 0u );

        // Check the actual strings themselves
        auto strings = std::unique_ptr<char*[]>{ new char*[stringCount] };
        for (size_t i = 0; i < stringCount; ++i) {
            ASSERT_GT( stringLengths[i], 0 );
            strings[i] = new char[stringLengths[i]+1];
            strings[i][stringLengths[i]] = 0;
        }
        EXPECT_EQ( stringList->to_host(strings.get(), 0, stringCount), 0 );
        EXPECT_STREQ( strings[0], "abc,\ndef, ghi" );
        EXPECT_STREQ( strings[1], "jkl, `mno`, pqr" );
        EXPECT_STREQ( strings[2], "stu `vwx` yz" );
        for (size_t i = 0; i < stringCount; ++i) {
            delete[] strings[i];
        }
    }
}

TEST(gdf_csv_test, IgnoreQuotes)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvIgnoreQuotesTest.csv";
    const char* names[]	= { "line", "verse" };
    const char* types[]	= { "int32", "str" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << "20,\"jkl \"\"mno\"\" pqr\"" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.quotechar = '\"';
        args.quoting = QUOTE_NONE;    // disable quoting
        args.doublequote = false;     // do not replace double quotechar with single
        args.skip_blank_lines = true;
        args.header = 0;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        // No filtering of any columns
        EXPECT_EQ( args.num_cols_out, args.num_names );

        // Check the parsed string column metadata
        ASSERT_EQ( args.data[1]->dtype, GDF_STRING );
        auto stringList = reinterpret_cast<NVStrings*>(args.data[1]->data);

        ASSERT_NE( stringList, nullptr );
        auto stringCount = stringList->size();
        ASSERT_EQ( stringCount, 3u );
        auto stringLengths = std::unique_ptr<int[]>{ new int[stringCount] };
        ASSERT_NE( stringList->byte_count(stringLengths.get(), false), 0u );

        // Check the actual strings themselves
        auto strings = std::unique_ptr<char*[]>{ new char*[stringCount] };
        for (size_t i = 0; i < stringCount; ++i) {
            ASSERT_GT( stringLengths[i], 0 );
            strings[i] = new char[stringLengths[i] + 1];
            strings[i][stringLengths[i]] = 0;
        }
        EXPECT_EQ( stringList->to_host(strings.get(), 0, stringCount), 0 );
        EXPECT_STREQ( strings[0], "\"abcdef ghi\"" );
        EXPECT_STREQ( strings[1], "\"jkl \"\"mno\"\" pqr\"" );
        EXPECT_STREQ( strings[2], "stu \"vwx\" yz" );
        for (size_t i = 0; i < stringCount; ++i) {
            delete[] strings[i];
        }
    }
}

TEST(gdf_csv_test, Booleans)
{
    const std::string fname = temp_env->get_temp_dir() + "CsvBooleansTest.csv";
    const char* names[] = {"A", "B", "C", "D"};
    const char* types[] = {"int32", "int32", "short", "bool"};
    const char* trueValues[] = {"yes", "Yes", "YES", "foo", "FOO"};
    const char* falseValues[] = {"no", "No", "NO", "Bar", "bar"};

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
              "true\nYes,5,foo,false\n";
    outfile.close();
    ASSERT_TRUE(checkFile(fname));

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.skip_blank_lines = true;
        args.true_values = trueValues;
        args.num_true_values = std::extent<decltype(trueValues)>::value;
        args.false_values = falseValues;
        args.num_false_values = std::extent<decltype(falseValues)>::value;
        args.header = -1;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        // Booleans are the same (integer) data type, but valued at 0 or 1
        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );
        ASSERT_EQ( args.data[2]->dtype, GDF_INT16 );
        ASSERT_EQ( args.data[3]->dtype, GDF_BOOL8 );

        auto firstCol = gdf_host_column<int32_t>(args.data[0]);
        EXPECT_THAT(firstCol.hostdata(), ::testing::ElementsAre(1, 0, 0, 0, 1));
        auto thirdCol = gdf_host_column<int16_t>(args.data[2]);
        EXPECT_THAT(thirdCol.hostdata(), ::testing::ElementsAre(0, 1, 1, 0, 1));
        auto fourthCol = gdf_host_column<cudf::bool8>(args.data[3]);
        EXPECT_THAT(
            fourthCol.hostdata(),
            ::testing::ElementsAre(cudf::true_v, cudf::true_v, cudf::false_v,
                                  cudf::true_v, cudf::false_v));
    }
}

TEST(gdf_csv_test, Dates)
{
    const std::string fname			= temp_env->get_temp_dir()+"CsvDatesTest.csv";
    const char* names[]			= { "A" };
    const char* types[]			= { "date" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.dayfirst = true;
        args.skip_blank_lines = true;
        args.header = -1;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_DATE64 );

        auto ACol = gdf_host_column<uint64_t>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(),
          ::testing::ElementsAre(983750400000, 1288483200000, 782611200000,
                       656208000000, 0, 798163200000, 774144000000,
                       1149679230400, 1126875750400, 2764800000) );
    }
}

TEST(gdf_csv_test, FloatingPoint)
{
    const std::string fname			= temp_env->get_temp_dir()+"CsvFloatingPoint.csv";
    const char* names[]			= { "A" };
    const char* types[]			= { "float32" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73.98007199999998;";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.decimal = '.';
        args.delimiter = ',';
        args.lineterminator = ';';
        args.skip_blank_lines = true;
        args.header = -1;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_FLOAT32 );

        auto ACol = gdf_host_column<float>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(),
            ::testing::Pointwise(FloatNearPointwise(1e-6),
                std::vector<float>{ 5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31, -73.98007199999998 }) );
    }
}

TEST(gdf_csv_test, Category)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvCategory.csv";
    const char* names[] = { "UserID" };
    const char* types[] = { "category" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "HBM0676;KRC0842;ILM1441;EJV0094;";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = ';';
        args.header = -1;
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_CATEGORY );

        auto ACol = gdf_host_column<int32_t>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(),
            ::testing::ElementsAre(2022314536, -189888986, 1512937027, 397836265) );
    }
}

TEST(gdf_csv_test, SkiprowsNrows)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvSkiprowsNrows.csv";
    const char* names[] = { "A" };
    const char* types[] = { "int32" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.skip_blank_lines = true;
        args.header = 1;
        args.skiprows = 2;
        args.nrows = 2;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(5, 6) );
    }
}

TEST(gdf_csv_test, ByteRange)
{
    const std::string fname = temp_env->get_temp_dir()+"CsvByteRange.csv";
    const char* names[] = { "A" };
    const char* types[] = { "int32" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.skip_blank_lines = true;
        args.header = -1;
        args.nrows = -1;
        args.byte_range_offset = 11;
        args.byte_range_size = 15;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(4000, 5000, 6000) );
    }
}

TEST(gdf_csv_test, BlanksAndComments)
{
    const std::string fname = temp_env->get_temp_dir()+"BlanksAndComments.csv";
    const char* names[] = { "A" };
    const char* types[] = { "int32" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
    outfile.close();
    ASSERT_TRUE( checkFile(fname) );

    {
        csv_read_arg args{};
        args.input_data_form = gdf_csv_input_form::FILE_PATH;
        args.filepath_or_buffer = fname.c_str();
        args.num_names = std::extent<decltype(names)>::value;
        args.names = names;
        args.num_dtype = args.num_names;
        args.dtype = types;
        args.delimiter = ',';
        args.lineterminator = '\n';
        args.skip_blank_lines = true;
        args.header = -1;
        args.comment = '#';
        args.nrows = -1;
        EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

        EXPECT_EQ( args.num_cols_out, args.num_names );
        ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );

        auto ACol = gdf_host_column<int32_t>(args.data[0]);
        EXPECT_THAT( ACol.hostdata(), ::testing::ElementsAre(1, 3, 4, 5, 8, 9) );
    }
}

TEST(gdf_csv_test, Writer)
{
    const std::string fname	= temp_env->get_temp_dir()+"CsvWriteTest.csv";
    const char* names[] = { "boolean", "integer", "float", "string" };
    const char* types[] = { "bool", "int32", "float32", "str" };

    std::ofstream outfile(fname, std::ofstream::out);
    outfile << "true,1,1.0,one" << '\n';
    outfile << "false,2,2.25,two" << '\n';
    outfile << "false,3,3.50,three" << '\n';
    outfile << "true,4,4.75,four" << '\n';
    outfile << "false,5,5.0,five" << '\n';
    outfile.close();

    csv_read_arg rargs{};
    rargs.input_data_form = gdf_csv_input_form::FILE_PATH;
    rargs.filepath_or_buffer = fname.c_str();
    rargs.num_names = std::extent<decltype(names)>::value;
    rargs.names = names;
    rargs.num_dtype = rargs.num_names;
    rargs.dtype = types;
    rargs.decimal = '.';
    rargs.delimiter = ',';
    rargs.lineterminator = '\n';
    rargs.skip_blank_lines = true;
    rargs.header = -1;
    rargs.nrows = -1;
    EXPECT_EQ( read_csv(&rargs), GDF_SUCCESS );

    const std::string ofname = temp_env->get_temp_dir()+"CsvWriteTestOut.csv";
    csv_write_arg wargs{};
    wargs.columns = rargs.data;  // columns from reader above
    wargs.filepath = ofname.c_str();
    wargs.num_cols = rargs.num_cols_out;
    wargs.delimiter = ',';
    wargs.line_terminator = "\n";

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
