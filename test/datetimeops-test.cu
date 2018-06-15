#include <cstdlib>
#include <iostream>

#include "gtest/gtest.h"
#include "../include/gdf/cffi/functions.h"


TEST(gdf_extract_datetime_year_TEST, date64Tests) {

	{

		gdf_column intputCol;


		thrust::device_vector<long long> leftSide(100);
		thrust::device_vector<long long> rightSide(100);

		for(int i = 0; i < leftSide.size(); i++){
			leftSide[i] = i;
			rightSide[i] = 4;
		}

		thrust::device_vector<long long> results(100);

		blazingArithmetic<thrust::device_vector<long long>::iterator, thrust::device_vector<long long>::iterator, thrust::device_vector<long long>::iterator>(leftSide.begin(), rightSide.begin(), results.begin(), "+", leftSide.size() );


		thrust::device_vector<long long> resultsTest(100);
		for(int i = 0; i < resultsTest.size(); i++){
			resultsTest[i] = i + 4;
		}

		bool same = true;
		for(int i = 0; i < resultsTest.size(); i++){
			if(results[i] != resultsTest[i]){
				same = false;
			}
		}


		EXPECT_TRUE( same == true );
	}
}
