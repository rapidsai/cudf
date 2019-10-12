#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestAttrs : public GdfTest{};

std::vector<const char*> hstrs{
        "Héllo", "thesé", nullptr, "ARE THE", "tést strings", "",
        "1.75", "-34", "+9.8", "17¼", "x³", "2³", " 12⅝",
        "1234567890", "de", "\t\r\n\f "};

TEST_F(TestAttrs, CharCounts)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<int> lengths(hstrs.size(),0);
    strs->len(lengths.data().get());

    int expected[] = { 5, 5, -1, 7, 12, 0, 4, 3, 4, 3, 2, 2, 4, 10, 2, 5};
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(lengths[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, ByteCounts)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<int> lengths(hstrs.size(),0);
    strs->byte_count(lengths.data().get());

    int expected[] = { 6, 6, -1, 7, 13, 0, 4, 3, 4, 4, 3, 3, 6, 10, 2, 5};
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(lengths[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, IsAlpha)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);
    strs->isalnum(results.data().get());
    {
        bool expected[] = { true, true, false, false, false, false,
                            false, false, false, true, true, true, false,
                            true, true, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->isalpha(results.data().get());
    {
        bool expected[] = { true, true, false, false, false, false,
                            false, false, false, false, false, false, false,
                            false, true, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->isspace(results.data().get());
    {
        bool expected[] = { false, false, false, false, false, false,
                            false, false, false, false, false, false, false,
                            false, false, true };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, IsNumeric)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);
    strs->isdigit(results.data().get());
    {
        bool expected[] = { false, false, false, false, false, false,
                            false, false, false, false, false, true, false,
                            true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->isdecimal(results.data().get());
    {
        bool expected[] = { false, false, false, false, false, false,
                            false, false, false, false, false, false, false,
                            true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->isnumeric(results.data().get());
    {
        bool expected[] = { false, false, false, false, false, false,
                            false, false, false, true, false, true, false,
                            true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, IsSpace)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);

    strs->isspace(results.data().get());
    {
        bool expected[] = { false, false, false, false, false, false,
                            false, false, false, false, false, false, false,
                            false, false, true };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->is_empty(results.data().get());
    {
        bool expected[] = { false, false, true, false, false, true,
                            false, false, false, false, false, false, false,
                            false, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, IsUpperLower)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),false);
    strs->isupper(results.data().get());
    {
        bool expected[] = { false, false, false, true, false, false,
                            true, true, true, true, false, true, true,
                            true, false, true };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->islower(results.data().get());
    {
        bool expected[] = { false, true, false, false, true, false,
                            true, true, true, true, true, true, true,
                            true, true, true };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestAttrs, CodePoints)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), 2);
    thrust::device_vector<unsigned int> results(10,0);
    strs->code_points(results.data().get());
    unsigned int expected[] = {72, 50089, 108, 108, 111, 116, 104, 101, 115, 50089};
    for( int idx = 0; idx < 10; ++idx )
        EXPECT_EQ(results[idx],expected[idx]);
}
