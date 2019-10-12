#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestCombine : public GdfTest{};

std::vector<const char*> hstrs1{ "thesé", nullptr, "are", "the",
                                 "tést", "strings", "" };
std::vector<const char*> hstrs2{ "1234", "accénted", "", nullptr, 
                                 "5678", "othér", "9" };

TEST_F(TestCombine, Concatenate)
{
    NVStrings* strs1 = NVStrings::create_from_array(hstrs1.data(),hstrs1.size());
    NVStrings* strs2 = NVStrings::create_from_array(hstrs2.data(),hstrs2.size());

    {    
        NVStrings* got = strs1->cat(strs2, nullptr);
        const char* expected[] = { "thesé1234", nullptr, "are", nullptr,
                                   "tést5678", "stringsothér", "9" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs1->cat(strs2, ":");
        const char* expected[] = { "thesé:1234", nullptr, "are:", nullptr,
                                   "tést:5678", "strings:othér", ":9" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs1->cat(strs2, ":", "_");
        const char* expected[] = { "thesé:1234", "_:accénted", "are:", "the:_",
                                   "tést:5678", "strings:othér", ":9" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs1);
    NVStrings::destroy(strs2);
}

std::vector<const char*> hstrs3{ "abcdéf", "", nullptr, "ghijkl",
                                 "mnop", "éach", "xyz" };

TEST_F(TestCombine, ConcatenateMultiple)
{
    NVStrings* strs1 = NVStrings::create_from_array(hstrs1.data(),hstrs1.size());
    NVStrings* strs2 = NVStrings::create_from_array(hstrs2.data(),hstrs2.size());
    NVStrings* strs3 = NVStrings::create_from_array(hstrs3.data(),hstrs2.size());

    std::vector<NVStrings*> others;
    others.push_back(strs2);
    others.push_back(strs3);
    {
        NVStrings* got = strs1->cat(others, nullptr);
        const char* expected[] = { "thesé1234abcdéf", nullptr, nullptr, nullptr,
                                   "tést5678mnop", "stringsothéréach", "9xyz" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs1->cat(others, ":");
        const char* expected[] = { "thesé:1234:abcdéf", nullptr, nullptr, nullptr,
                                   "tést:5678:mnop", "strings:othér:éach", ":9:xyz" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs1->cat(others, ":", "_");
        const char* expected[] = { "thesé:1234:abcdéf", "_:accénted:", "are::_", "the:_:ghijkl",
                                   "tést:5678:mnop", "strings:othér:éach", ":9:xyz" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs1);
    NVStrings::destroy(strs2);
    NVStrings::destroy(strs3);
}

TEST_F(TestCombine, Join)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs1.data(),hstrs1.size());

    {    
        NVStrings* got = strs->join("");
        const char* expected[] = { "theséarethetéststrings" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs->join(":");
        const char* expected[] = { "thesé:are:the:tést:strings:" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    {    
        NVStrings* got = strs->join(":", "_");
        const char* expected[] = { "thesé:_:are:the:tést:strings:" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}
