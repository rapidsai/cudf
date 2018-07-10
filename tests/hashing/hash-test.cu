#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

struct gdf_hashing_test : public ::testing::Test {

	void TearDown() {

	}
};

TEST(gdf_hashing_test, simpleTest) {

  EXPECT_TRUE( 1+1 == 2 );

}

