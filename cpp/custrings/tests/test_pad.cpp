#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestPad : public GdfTest{};

std::vector<const char*> hstrs{ "12345", "thesé", nullptr, "ARE THE", "tést strings", "" };

TEST_F(TestPad, Repeat)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->repeat(1);
        const char* expected[] = { "12345", "thesé", nullptr, "ARE THE", "tést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->repeat(2);
        const char* expected[] = { "1234512345", "theséthesé", nullptr, "ARE THEARE THE", "tést stringstést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestPad, Pad)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->ljust(10);
        const char* expected[] = { "12345     ", "thesé     ", nullptr, "ARE THE   ", "tést strings", "          " };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->rjust(7);
        const char* expected[] = { "  12345", "  thesé", nullptr, "ARE THE", "tést strings", "       " };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->center(9, "_");
        const char* expected[] = { "__12345__", "__thesé__", nullptr, "_ARE THE_", "tést strings", "_________" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestPad, ZFill)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    NVStrings* got = strs->zfill(8);
    const char* expected[] = { "00012345", "000thesé", nullptr, "0ARE THE", "tést strings", "00000000" };
    EXPECT_TRUE( verify_strings(got,expected) );
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestPad, Wrap)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    NVStrings* got = strs->wrap(3);
    const char* expected[] = { "12345", "thesé", nullptr, "ARE\nTHE", "tést\nstrings", "" };
    EXPECT_TRUE( verify_strings(got,expected) );
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}
