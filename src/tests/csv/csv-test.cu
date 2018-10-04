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
#include <vector>
#include <sys/stat.h>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
 
struct gdf_csv_test : public ::testing::Test {
  void TearDown() {
  }
};
 
bool checkFile(const char *fpath) {
	struct stat     st;

	if (stat(fpath, &st)) {
		return 0;
	}
	return 1;
}

TEST(gdf_csv_test, CsvSimple)
{
	gdf_error error = GDF_SUCCESS;

	csv_read_arg	args;

    args.num_cols = 10;

    args.names = new const char*[10] {
    	"A",
    	"B",
    	"C",
    	"D",
    	"E",
    	"F",
    	"G",
    	"H",
    	"I",
    	"J"
    };

    args.dtype = new const char *[10]{
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32",
    		"int32"
    };


	args.file_path = (char *)("/tmp/simple.csv");

	if (  checkFile(args.file_path)) {

		args.delimiter 		= ',';
		args.lineterminator = '\n';
		args.delim_whitespace = 0;
		args.skipinitialspace = 0;
		args.skiprows 		= 0;
		args.skipfooter 	= 0;
		args.dayfirst 		= 0;

		error = read_csv(&args);
	}

	EXPECT_TRUE( error == GDF_SUCCESS );
}



TEST(gdf_csv_test, MortPerf)
{
	gdf_error error = GDF_SUCCESS;

	csv_read_arg	args;
	const int num_cols = 31;

    args.num_cols = num_cols;

    args.names = new const char *[num_cols] {
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

  args.dtype = new const char *[num_cols] {
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



	args.file_path = (char *)("/tmp/Performance_2000Q1.txt");

	if (  checkFile(args.file_path))
	{
		args.delimiter 		= '|';
		args.lineterminator = '\n';
		args.delim_whitespace = 0;
		args.skipinitialspace = 0;
		args.skiprows 		= 0;
		args.skipfooter 	= 0;
		args.dayfirst 		= 0;

		error = read_csv(&args);
	}

	EXPECT_TRUE( error == GDF_SUCCESS );
}



