#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"
#include "./utils.h"

std::vector<const char*> hstrs{ "Héllo", "thesé", nullptr, "ARE THE", "tést strings", "" };

TEST(TestFind, Compare)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<int> results(hstrs.size(),0);
    strs->compare("thesé",results.data().get());

    int expected[] = { -44,0,-1,-51,-165,-1 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}


TEST(TestFind, Find)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<int> results(hstrs.size(),0);
    strs->find("é",0,-1,results.data().get());
    {
        int expected[] = { 1,4,-2,-1,1,-1 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    strs->rfind("l",0,-1,results.data().get());
    {
        int expected[] = { 3,-1,-2,-1,-1,-1 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    NVStrings::destroy(strs);
}

TEST(TestFind, Match)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    std::vector<const char*> hstrs2{ "héllo", "thesé", "", nullptr, "tést strings", "" };
    NVStrings* strs2 = NVStrings::create_from_array(hstrs2.data(), hstrs2.size());

    thrust::device_vector<bool> results(hstrs.size(),false);
    strs->match_strings(*strs2,results.data().get());
    {
        bool expected[] = { false, true, false, false, true, true };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    NVStrings::destroy(strs);
    NVStrings::destroy(strs2);
}

TEST(TestFind, Contains)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),0);
    strs->contains("s",results.data().get());

    bool expected[] = { false, true, false, false, true, false };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST(TestFind, FindFrom)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    thrust::device_vector<int> from(hstrs.size(),3);
    thrust::device_vector<int> results(hstrs.size(),0);
    strs->find_from("s",from.data().get(),nullptr,results.data().get());
    {
        int expected[] = { -1,3,-2,-1,5,-1 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST(TestFind, FindMultiple)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    std::vector<const char*> hstrs2{ "é", "e" };
    NVStrings* strs2 = NVStrings::create_from_array(hstrs2.data(), hstrs2.size());

    thrust::device_vector<int> results(hstrs.size()*hstrs2.size(),0);
    strs->find_multiple(*strs2,results.data().get());
    {
        int expected[] = { 1,-1, 4,2, -2,-2, -1,-1, 1,-1, -1,-1 };
        for( int idx = 0; idx < (int) (hstrs.size() * hstrs2.size()); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
    NVStrings::destroy(strs2);
}

TEST(TestFind, StartsEnds)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),0);
    {
        strs->endswith("E",results.data().get());
        bool expected[] = { false, false, false, true, false, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    {
        strs->startswith("t",results.data().get());
        bool expected[] = { false, true, false, false, true, false };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST(TestFind, FindAll)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    std::vector<NVStrings*> results;
    strs->findall("(\\w+)", results);
    ASSERT_EQ( (int) results.size(), 2 );
    const char* expected1[] = { "Héllo", "thesé", nullptr, "ARE", "tést", nullptr };
    EXPECT_TRUE( verify_strings(results[0],expected1));
    const char* expected2[] = { nullptr, nullptr, nullptr, "THE", "strings", nullptr };
    EXPECT_TRUE( verify_strings(results[1],expected2));

    for( auto itr = results.begin(); itr != results.end(); itr++ )
        NVStrings::destroy(*itr);
    NVStrings::destroy(strs);
}

TEST(TestFind, FindAllRecord)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    std::vector<NVStrings*> results;
    strs->findall_record("(\\w+)", results);
    ASSERT_EQ( results.size(), strs->size() );

    std::vector< std::vector<const char*> > expected;
    expected.push_back( std::vector<const char*>{"Héllo"} );
    expected.push_back( std::vector<const char*>{"thesé"} );
    expected.push_back( std::vector<const char*>{} );
    expected.push_back( std::vector<const char*>{"ARE","THE"} );
    expected.push_back( std::vector<const char*>{"tést","strings"} );
    expected.push_back( std::vector<const char*>{} );

    for( size_t idx = 0; idx < results.size(); ++idx )
    {
        NVStrings* row = results[idx];
        ASSERT_EQ( row->size(), expected[idx].size() );
        EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
        NVStrings::destroy(row);
    }
    NVStrings::destroy(strs);
}


int main( int argc, char** argv )
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}