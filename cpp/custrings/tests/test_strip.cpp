#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

std::vector<const char*> hstrs{ " hello  ", "   thesé ", nullptr, "ARE THE", " tést  strings ", "" };

TEST(TestStrip, Strip)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->lstrip(" ");
        const char* expected[] = { "hello  ", "thesé ", nullptr, "ARE THE", "tést  strings ", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->rstrip(" ");
        const char* expected[] = { " hello", "   thesé", nullptr, "ARE THE", " tést  strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->strip(" ");
        const char* expected[] = { "hello", "thesé", nullptr, "ARE THE", "tést  strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}


int main( int argc, char** argv )
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}