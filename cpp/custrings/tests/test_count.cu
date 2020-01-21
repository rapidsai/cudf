#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestCount : public GdfTest{};

std::vector<const char*> hstrs{
        "The quick brown @fox jumps", "ovér the", "lazy @dog",
        "1234", "00:0:00", nullptr, "" };

TEST_F(TestCount, Contains)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);

    {
        strs->contains("é", results.data().get());
        bool expected[] = { false, true, false, false, false, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->contains_re("\\d+", results.data().get());
        bool expected[] = { false, false, false, true, true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->contains_re("@\\w+", results.data().get());
        bool expected[] = { true, false, true, false, false, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestCount, Match)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);

    {
        strs->match("ov[eé]r", results.data().get());
        bool expected[] = { false, true, false, false, false, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->match("[tT]he", results.data().get());
        bool expected[] = { true, false, false, false, false, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->match("\\d+", results.data().get());
        bool expected[] = { false, false, false, true, true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestCount, Count)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<int> results(hstrs.size(),0);

    {
        strs->count_re("[tT]he", results.data().get());
        int expected[] = { 1, 1, 0,0,0,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->count_re("@\\w+", results.data().get());
        int expected[] = { 1, 0, 1, 0,0,0,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {
        strs->count_re("\\d+:\\d+", results.data().get());
        int expected[] = { 0,0,0, 0,1,0,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}
