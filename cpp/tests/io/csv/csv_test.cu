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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <cudf.h>
#include <NVStrings.h>

bool checkFile(const char *fname)
{
	struct stat st;
	return (stat(fname, &st) ? 0 : 1);
}

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
			std::cout << "[" << i << "]: value=" << m_hostdata[i] << "\n";
		}
	}

private:
	std::vector<T> m_hostdata;
};

TEST(gdf_csv_test, Simple)
{
	const char* fname	= "/tmp/CsvSimpleTest.csv";
	const char* names[]	= { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J" };
	const char* types[]	= { "int32", "int32", "int32", "int32", "int32",
							"int32", "int32", "int32", "int32", "int32", };

	std::ofstream outfile(fname, std::ofstream::out);
	outfile <<	"10,20,30,40,50,60,70,80,90,100\n"\
				"11,21,31,41,51,61,71,81,91,101\n"\
				"12,22,32,42,52,62,72,82,92,102\n"\
				"13,23,33,43,53,63,73,83,93,103\n";
	outfile.close();
	ASSERT_TRUE( checkFile(fname) );

	{
		csv_read_arg args{};
		args.input_data_form = gdf_csv_input_form::FILE_PATH;
		args.filepath_or_buffer = fname;
		args.num_cols		= std::extent<decltype(names)>::value;
		args.names			= names;
		args.dtype			= types;
		args.delimiter		= ',';
		args.lineterminator = '\n';
		args.nrows = -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		auto firstCol = gdf_host_column<int32_t>(args.data[0]);
		auto sixthCol = gdf_host_column<int32_t>(args.data[5]);
		EXPECT_THAT(firstCol.hostdata(), ::testing::ElementsAre(10, 11, 12, 13));
		EXPECT_THAT(sixthCol.hostdata(), ::testing::ElementsAre(60, 61, 62, 63));
	}
}

TEST(gdf_csv_test, MortPerf)
{
	gdf_error error = GDF_SUCCESS;

	csv_read_arg	args{};
	const int num_cols = 31;

    args.num_cols = num_cols;
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
		args.filepath_or_buffer = (char *)("/tmp/Performance_2000Q1.txt");

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
	const char* fname	= "/tmp/CsvStringsTest.csv";
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
		args.filepath_or_buffer = fname;
		args.num_cols		= std::extent<decltype(names)>::value;
		args.names			= names;
		args.dtype			= types;
		args.delimiter		= ',';
		args.lineterminator = '\n';
		args.skiprows		= 1;
		args.nrows = -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		// No filtering of any columns
		EXPECT_EQ( args.num_cols_out, args.num_cols );

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
			strings[i] = new char[stringLengths[i]];
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
	const char* fname	= "/tmp/CsvQuotedStringsTest.csv";
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
		args.filepath_or_buffer = fname;
		args.num_cols		= std::extent<decltype(names)>::value;
		args.names			= names;
		args.dtype			= types;
		args.delimiter		= ',';
		args.lineterminator = '\n';
		args.quotechar		= '`';
		args.quoting		= true;	// strip outermost quotechar
		args.doublequote	= true;	// replace double quotechar with single
		args.skiprows		= 1;
		args.nrows = -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		// No filtering of any columns
		EXPECT_EQ( args.num_cols_out, args.num_cols );

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
			strings[i] = new char[stringLengths[i]];
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

TEST(gdf_csv_test, KeepFullQuotedStrings)
{
	const char* fname	= "/tmp/CsvKeepFullQuotedStringsTest.csv";
	const char* names[]	= { "line", "verse" };
	const char* types[]	= { "int32", "str" };

	std::ofstream outfile(fname, std::ofstream::out);
	outfile << names[0] << ',' << names[1] << ',' << '\n';
	outfile << "10,\"abc,\ndef, ghi\"" << '\n';
	outfile << "20,\"jkl, \"\"mno\"\", pqr\"" << '\n';
	outfile << "30,stu \"vwx\" yz" << '\n';
	outfile.close();
	ASSERT_TRUE( checkFile(fname) );

	{
		csv_read_arg args{};
		args.input_data_form = gdf_csv_input_form::FILE_PATH;
		args.filepath_or_buffer = fname;
		args.num_cols		= std::extent<decltype(names)>::value;
		args.names			= names;
		args.dtype			= types;
		args.delimiter		= ',';
		args.lineterminator = '\n';
		args.quotechar		= '\"';
		args.quoting		= false;	// do not strip outermost quotechar
		args.doublequote	= false;	// do not replace double quotechar with single
		args.skiprows		= 1;
		args.nrows = -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		// No filtering of any columns
		EXPECT_EQ( args.num_cols_out, args.num_cols );

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
			strings[i] = new char[stringLengths[i]];
		}
		EXPECT_EQ( stringList->to_host(strings.get(), 0, stringCount), 0 );
		EXPECT_STREQ( strings[0], "\"abc,\ndef, ghi\"" );
		EXPECT_STREQ( strings[1], "\"jkl, \"\"mno\"\", pqr\"" );
		EXPECT_STREQ( strings[2], "stu \"vwx\" yz" );
		for (size_t i = 0; i < stringCount; ++i) {
			delete[] strings[i];
		}
	}
}

TEST(gdf_csv_test, SpecifiedBoolValues)
{
	const char* fname			= "/tmp/CsvSpecifiedBoolValuesTest.csv";
	const char* names[]			= { "A", "B", "C" };
	const char* types[]			= { "int32", "int32", "short" };
	const char* trueValues[]	= { "yes", "Yes", "YES", "foo", "FOO" };
	const char* falseValues[]	= { "no", "No", "NO", "Bar", "bar" };

	std::ofstream outfile(fname, std::ofstream::out);
	outfile << "YES,1,bar\nno,2,FOO\nBar,3,yes\nNo,4,NO\nYes,5,foo\n";
	outfile.close();
	ASSERT_TRUE( checkFile(fname) );

	{
		csv_read_arg args{};
		args.input_data_form		= gdf_csv_input_form::FILE_PATH;
		args.filepath_or_buffer		= fname;
		args.num_cols			= std::extent<decltype(names)>::value;
		args.names				= names;
		args.dtype				= types;
		args.delimiter			= ',';
		args.lineterminator 	= '\n';
		args.true_values		= trueValues;
		args.num_true_values	= std::extent<decltype(trueValues)>::value;
		args.false_values		= falseValues;
		args.num_false_values	= std::extent<decltype(falseValues)>::value;
		args.nrows = -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		// Booleans are the same (integer) data type, but valued at 0 or 1
		EXPECT_EQ( args.num_cols_out, args.num_cols );
		ASSERT_EQ( args.data[0]->dtype, GDF_INT32 );
		ASSERT_EQ( args.data[2]->dtype, GDF_INT16 );

		auto firstCol = gdf_host_column<int32_t>(args.data[0]);
		EXPECT_THAT(firstCol.hostdata(), ::testing::ElementsAre(1, 0, 0, 0, 1));
		auto thirdCol = gdf_host_column<int16_t>(args.data[2]);
		EXPECT_THAT(thirdCol.hostdata(), ::testing::ElementsAre(0, 1, 1, 0, 1));
	}
}

TEST(gdf_csv_test, Dates)
{
	const char* fname			= "/tmp/CsvDatesTest.csv";
	const char* names[]			= { "A" };
	const char* types[]			= { "date" };

	std::ofstream outfile(fname, std::ofstream::out);
	outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
	outfile << "18/04/1995\n14/07/1994\n07/06/2006\n16/09/2005\n2/2/1970\n";
	outfile.close();
	ASSERT_TRUE( checkFile(fname) );

	{
		csv_read_arg args{};
		args.input_data_form	= gdf_csv_input_form::FILE_PATH;
		args.filepath_or_buffer	= fname;
		args.num_cols			= std::extent<decltype(names)>::value;
		args.names				= names;
		args.dtype				= types;
		args.delimiter			= ',';
		args.lineterminator 	= '\n';
		args.dayfirst			= true;
		args.nrows				= -1;
		EXPECT_EQ( read_csv(&args), GDF_SUCCESS );

		EXPECT_EQ( args.num_cols_out, args.num_cols );
		ASSERT_EQ( args.data[0]->dtype, GDF_DATE64 );

		auto ACol = gdf_host_column<uint64_t>(args.data[0]);
		EXPECT_THAT( ACol.hostdata(),
			::testing::ElementsAre(983750400000, 1288483200000, 782611200000,
								   656208000000, 0, 798163200000, 774144000000,
								   1149638400000, 1126828800000, 2764800000) );
	}
}
