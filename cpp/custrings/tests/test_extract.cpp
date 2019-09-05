#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

 std::vector<const char*> hstrs{"First Last", "Joe Schmoe", "John Smith", "Jane Smith",
                                "Beyonce", "Sting",
                                nullptr, "" };

TEST(TestExtract, ExtractColumn)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    std::vector<NVStrings*> results;
    strs->extract("(\\w+) (\\w+)", results);
    ASSERT_EQ( results.size(), 2 );
    const char* expected1[] = { "First", "Joe", "John", "Jane", nullptr, nullptr, nullptr, nullptr };
    EXPECT_TRUE( verify_strings(results[0],expected1));
    const char* expected2[] = { "Last", "Schmoe", "Smith", "Smith", nullptr, nullptr, nullptr, nullptr };
    EXPECT_TRUE( verify_strings(results[1],expected2));

    for( auto itr=results.begin(); itr!=results.end(); itr++ )
        NVStrings::destroy(*itr);
    NVStrings::destroy(strs);
}

TEST(TestExtract, ExtractRecord)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    std::vector<NVStrings*> results;
    strs->extract_record("(\\w+) (\\w+)", results);
    ASSERT_EQ( results.size(), strs->size() );

    std::vector< std::vector<const char*> > expected;
    expected.push_back( std::vector<const char*>{"First","Last"} );
    expected.push_back( std::vector<const char*>{"Joe","Schmoe"} );
    expected.push_back( std::vector<const char*>{"John","Smith"} );
    expected.push_back( std::vector<const char*>{"Jane","Smith"} );
    expected.push_back( std::vector<const char*>{nullptr,nullptr} );
    expected.push_back( std::vector<const char*>{nullptr,nullptr} );
    expected.push_back( std::vector<const char*>{nullptr,nullptr} );
    expected.push_back( std::vector<const char*>{nullptr,nullptr} );

    for( size_t idx=0; idx < results.size(); ++idx )
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