#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestSplit : public GdfTest{};

std::vector<const char*> hstrs{ "Héllo thesé", nullptr, "are some", "tést String", "" };


TEST_F(TestSplit, Split)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    {
        std::vector<NVStrings*> results;
        strs->split(-1, results);
        ASSERT_EQ( results.size(), 2 );
        const char* expected1[] = { "Héllo", nullptr, "are", "tést", nullptr };
        EXPECT_TRUE( verify_strings(results[0],expected1));
        const char* expected2[] = { "thesé", nullptr, "some", "String", nullptr };
        EXPECT_TRUE( verify_strings(results[1],expected2));
        for( auto itr=results.begin(); itr!=results.end(); itr++ )
            NVStrings::destroy(*itr);
    }
    {
        std::vector<NVStrings*> results;
        strs->rsplit(-1, results);
        ASSERT_EQ( results.size(), 2 );
        const char* expected1[] = { "Héllo", nullptr, "are", "tést", nullptr };
        EXPECT_TRUE( verify_strings(results[0],expected1));
        const char* expected2[] = { "thesé", nullptr, "some", "String", nullptr };
        EXPECT_TRUE( verify_strings(results[1],expected2));
        for( auto itr=results.begin(); itr!=results.end(); itr++ )
            NVStrings::destroy(*itr);
    }
    {
        std::vector<NVStrings*> results;
        strs->split("s", -1, results);
        ASSERT_EQ( results.size(), 2 );
        const char* expected1[] = { "Héllo the", nullptr, "are ", "té", "" };
        EXPECT_TRUE( verify_strings(results[0],expected1));
        const char* expected2[] = { "é", nullptr, "ome", "t String", nullptr };
        EXPECT_TRUE( verify_strings(results[1],expected2));
        for( auto itr=results.begin(); itr!=results.end(); itr++ )
            NVStrings::destroy(*itr);
    }
    {
        std::vector<NVStrings*> results;
        strs->rsplit("s", 2, results);
        ASSERT_EQ( results.size(), 2 );
        const char* expected1[] = { "Héllo the", nullptr, "are ", "té", "" };
        EXPECT_TRUE( verify_strings(results[0],expected1));
        const char* expected2[] = { "é", nullptr, "ome", "t String", nullptr };
        EXPECT_TRUE( verify_strings(results[1],expected2));
        for( auto itr=results.begin(); itr!=results.end(); itr++ )
            NVStrings::destroy(*itr);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestSplit, SplitRecord)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    {
        std::vector<NVStrings*> results;
        strs->split_record(-1,results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo","thesé"} );
        expected.push_back( std::vector<const char*>{} );
        expected.push_back( std::vector<const char*>{"are","some"} );
        expected.push_back( std::vector<const char*>{"tést","String"} );
        expected.push_back( std::vector<const char*>{""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    {
        std::vector<NVStrings*> results;
        strs->rsplit_record(-1,results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo","thesé"} );
        expected.push_back( std::vector<const char*>{} );
        expected.push_back( std::vector<const char*>{"are","some"} );
        expected.push_back( std::vector<const char*>{"tést","String"} );
        expected.push_back( std::vector<const char*>{""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    {
        std::vector<NVStrings*> results;
        strs->split_record("s",-1,results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo the","é"} );
        expected.push_back( std::vector<const char*>{} );
        expected.push_back( std::vector<const char*>{"are ","ome"} );
        expected.push_back( std::vector<const char*>{"té","t String"} );
        expected.push_back( std::vector<const char*>{""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    {
        std::vector<NVStrings*> results;
        strs->split_record("s",2,results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo the","é"} );
        expected.push_back( std::vector<const char*>{} );
        expected.push_back( std::vector<const char*>{"are ","ome"} );
        expected.push_back( std::vector<const char*>{"té","t String"} );
        expected.push_back( std::vector<const char*>{""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    NVStrings::destroy(strs);
}

TEST_F(TestSplit, Partition)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    {
        std::vector<NVStrings*> results;
        strs->partition(" ",results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo"," ", "thesé"} );
        expected.push_back( std::vector<const char*>{nullptr, nullptr, nullptr} );
        expected.push_back( std::vector<const char*>{"are"," ","some"} );
        expected.push_back( std::vector<const char*>{"tést"," ","String"} );
        expected.push_back( std::vector<const char*>{"","",""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    {
        std::vector<NVStrings*> results;
        strs->rpartition(" ",results);
        ASSERT_EQ( results.size(), strs->size() );

        std::vector< std::vector<const char*> > expected;
        expected.push_back( std::vector<const char*>{"Héllo"," ", "thesé"} );
        expected.push_back( std::vector<const char*>{nullptr, nullptr, nullptr} );
        expected.push_back( std::vector<const char*>{"are"," ","some"} );
        expected.push_back( std::vector<const char*>{"tést"," ","String"} );
        expected.push_back( std::vector<const char*>{"","",""} );

        for( size_t idx=0; idx < results.size(); ++idx )
        {
            NVStrings* row = results[idx];
            if( !row )
                continue;
            ASSERT_EQ( row->size(), expected[idx].size() );
            EXPECT_TRUE( verify_strings(row,expected[idx].data()) );
            NVStrings::destroy(row);
        }
    }
    NVStrings::destroy(strs);
}
