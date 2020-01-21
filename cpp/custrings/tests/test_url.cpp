#include <gtest/gtest.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestURL : public GdfTest{};

TEST_F(TestURL, UrlEncode)
{
    std::vector<const char*> hstrs{"www.nvidia.com/rapids?p=é", "/_file-7.txt", "a b+c~d",
                                   "e\tfgh\\jklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                   "0123456789", " \t\f\n",
                                   nullptr, "" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->url_encode();
    const char* expected[] = { "www.nvidia.com%2Frapids%3Fp%3D%C3%A9", "%2F_file-7.txt", "a%20b%2Bc~d",
                               "e%09fgh%5Cjklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                               "0123456789", "%20%09%0C%0A",
                               nullptr, "" };
    EXPECT_TRUE( verify_strings(got, expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestURL, UrlDecode)
{
    std::vector<const char*> hstrs{ "www.nvidia.com/rapids/%3Fp%3D%C3%A9", "/_file-1234567890.txt", "a%20b%2Bc~defghijklmnopqrstuvwxyz",
                                    "%25-accent%c3%a9d", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "01234567890",
                                    nullptr, "" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->url_decode();
    const char* expected[] = { "www.nvidia.com/rapids/?p=é", "/_file-1234567890.txt", "a b+c~defghijklmnopqrstuvwxyz",
                               "%-accentéd", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "01234567890",
                               nullptr, "" };

    EXPECT_TRUE( verify_strings(got, expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}
