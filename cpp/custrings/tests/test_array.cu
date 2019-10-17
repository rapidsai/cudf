#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestArray : public GdfTest{};

std::vector<const char*> hstrs{ "John Smith", "Joe Blow", "Jane Smith", nullptr, "" };

TEST_F(TestArray, Sublist)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->sublist(1,4);

    const char* expected[] = {"Joe Blow", "Jane Smith", nullptr };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, Gather)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<int> indexes;
    indexes.push_back(1);
    indexes.push_back(3);
    indexes.push_back(2);

    NVStrings* got = strs->gather(indexes.data().get(), indexes.size());
    const char* expected[] = {"Joe Blow", nullptr, "Jane Smith" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, GatherBool)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<bool> indexes;
    indexes.push_back(true);
    indexes.push_back(false);
    indexes.push_back(false);
    indexes.push_back(false);
    indexes.push_back(true);

    NVStrings* got = strs->gather(indexes.data().get());
    const char* expected[] = {"John Smith", "" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, Scatter)
{
    NVStrings* strs1 = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    const char* h2[] = { "", "Joe Schmoe" };
    NVStrings* strs2 = NVStrings::create_from_array(h2,2);

    thrust::device_vector<int> indexes;
    indexes.push_back(1);
    indexes.push_back(3);

    {
        NVStrings* got = strs1->scatter(*strs2, indexes.data().get() );
        const char* expected[] = {"John Smith", "", "Jane Smith", "Joe Schmoe", "" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs1->scatter("_", indexes.data().get(), indexes.size() );
        const char* expected[] = {"John Smith", "_", "Jane Smith", "_", "" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs1);
    NVStrings::destroy(strs2);
}

TEST_F(TestArray, RemoveStrings)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<int> indexes;
    indexes.push_back(0);
    indexes.push_back(3);
    indexes.push_back(2);

    NVStrings* got = strs->remove_strings(indexes.data().get(), indexes.size());
    const char* expected[] = { "Joe Blow", "" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, SortLength)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->sort(NVStrings::length);
    const char* expected[] = { nullptr, "", "Joe Blow", "John Smith", "Jane Smith" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, SortName)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->sort(NVStrings::name);
    const char* expected[] = { nullptr, "",  "Jane Smith", "Joe Blow", "John Smith" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestArray, OrderLength)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> indexes(strs->size());

    strs->order(NVStrings::length, false, indexes.data().get() );
    unsigned int expected[] = { 3,0,2,1,4 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(indexes[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestArray, OrderName)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> indexes(strs->size());

    strs->order(NVStrings::name, false, indexes.data().get(), false );
    unsigned int expected[] = { 0,1,2,4,3 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(indexes[idx],expected[idx]);

    NVStrings::destroy(strs);
}
