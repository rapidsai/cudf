#include <gtest/gtest.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestModify : public GdfTest{};

std::vector<const char*> hstrs{ "Héllo", "thesé", nullptr, "ARE THE", "tést strings", "" };

TEST_F(TestModify, SliceReplace)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->slice_replace("___",2,3);
        const char* expected[] = { "Hé___lo", "th___sé", nullptr, "AR___ THE", "té___t strings", "___" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->slice_replace("||",3,3);
        const char* expected[] = { "Hél||lo", "the||sé", nullptr, "ARE|| THE", "tés||t strings", "||" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->slice_replace("x",-1,-1);
        const char* expected[] = { "Héllox", "theséx", nullptr, "ARE THEx", "tést stringsx", "x" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestModify, Slice)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->slice(2,3);
        const char* expected[] = { "l", "e", nullptr, "E", "s", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->slice(3,-1);
        const char* expected[] = { "lo", "sé", nullptr, " THE", "t strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->slice(0,0);
        const char* expected[] = { "", "", nullptr, "", "", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->get(0);
        const char* expected[] = { "H", "t", nullptr, "A", "t", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestModify, SliceFrom)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    {
        thrust::device_vector<int> from(strs->size(),4);
        NVStrings* got = strs->slice_from(from.data().get());
        const char* expected[] = { "o", "é", nullptr, "THE", " strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        thrust::device_vector<int> starts(strs->size(),0);
        thrust::device_vector<int> ends(strs->size(),0);
        NVStrings* got = strs->slice_from(starts.data().get(),ends.data().get());
        const char* expected[] = { "", "", nullptr, "", "", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    NVStrings::destroy(strs);
}

TEST_F(TestModify, FillNa)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());
    {
        NVStrings* got = strs->fillna("||");
        const char* expected[] = { "Héllo", "thesé", "||", "ARE THE", "tést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        std::vector<const char*> nas{ "1", "2", "3", "4", "5", "6" };
        NVStrings* dnas = NVStrings::create_from_array(nas.data(),nas.size());
        NVStrings* got = strs->fillna(*dnas);
        const char* expected[] = { "Héllo", "thesé", "3", "ARE THE", "tést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
        NVStrings::destroy(dnas);
    }
    NVStrings::destroy(strs);
}

TEST_F(TestModify, Insert)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    {
        NVStrings* got = strs->insert("***",1);
        const char* expected[] = { "H***éllo", "t***hesé", nullptr, "A***RE THE", "t***ést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = strs->insert("++",-1);
        const char* expected[] = { "Héllo++", "thesé++", nullptr, "ARE THE++", "tést strings++", "++" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestModify, Translate)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(), hstrs.size());

    std::vector< std::pair<unsigned,unsigned> > table;
    table.push_back( std::pair<unsigned,unsigned>{'e','E'});
    table.push_back( std::pair<unsigned,unsigned>{'H','h'});

    {
        NVStrings* got = strs->translate(table.data(),table.size());
        const char* expected[] = { "héllo", "thEsé", nullptr, "ARE ThE", "tést strings", "" };
        EXPECT_TRUE( verify_strings(got,expected) );
        NVStrings::destroy(got);
    }

    NVStrings::destroy(strs);
}
