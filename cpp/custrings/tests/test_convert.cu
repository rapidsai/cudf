
#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestConvert : public GdfTest{};

TEST_F(TestConvert, Hash)
{
    std::vector<const char*> hstrs{ "thesé", nullptr, "are", "the",
                                "tést", "strings", "" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> results(hstrs.size(),0);
    strs->hash(results.data().get());
    unsigned int expected[] = { 126208335, 0, 3771471008, 2967174367, 1378466566,
                                3184694146, 1257683291 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestConvert, ToInteger)
{
    std::vector<const char*> hstrs{"1234", nullptr, 
            "-876", "543.2", "-0.12", ".55", "-.002",
            "", "de", "abc123", "123abc", "456e", "-1.78e+5"};
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    {    
        thrust::device_vector<int> results(hstrs.size(),0);
        strs->stoi(results.data().get());
        int expected[] = { 1234, 0,
            -876, 543, 0, 0, 0,
            0, 0, 0, 123, 456, -1 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    {    
        thrust::device_vector<long> results(hstrs.size(),0);
        strs->stol(results.data().get());
        long expected[] = { 1234, 0,
            -876, 543, 0, 0, 0,
            0, 0, 0, 123, 456, -1 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestConvert, FromInteger)
{
    {    
        int values[] = {100, 987654321, -12761, 0, 5, -4};
        thrust::device_vector<int> results(6);
        cudaMemcpy( results.data().get(), values, 6*sizeof(int), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::itos(results.data().get(),6);
        const char* expected[] = { "100", "987654321", "-12761", "0", "5", "-4" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    {    
        long values[] = {100000, 9876543210, -1276100, 0, 5, -4};
        thrust::device_vector<long> results(6);
        cudaMemcpy( results.data().get(), values, 6*sizeof(long), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::ltos(results.data().get(),6);
        const char* expected[] = { "100000", "9876543210", "-1276100", "0", "5", "-4" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
}

TEST_F(TestConvert, Hex)
{
    std::vector<const char*> hstrs{"1234", nullptr, 
            "98BEEF", "1a5", "CAFE", "2face"};
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    thrust::device_vector<unsigned int> results(hstrs.size(),0);
    strs->htoi(results.data().get());
    unsigned int expected[] = { 4660, 0,
        10010351, 421, 51966, 195278 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);
    NVStrings::destroy(strs);
}

TEST_F(TestConvert, ToFloat)
{
    std::vector<const char*> hstrs{"1234", nullptr, 
            "-876", "543.2", "-0.12", ".25", "-.002",
            "", "NaN", "abc123", "123abc", "456e", "-1.78e+5",
            "-122.33644782123456789", "12e+309" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    {    
        float nanval = std::numeric_limits<float>::quiet_NaN();
        float infval = std::numeric_limits<float>::infinity();
        thrust::device_vector<float> results(hstrs.size(),0);
        strs->stof(results.data().get());
        float expected[] = { 1234.0, 0,
            -876.0, 543.2, -0.12, 0.25, -0.002,
            0, nanval, 0, 123.0, 456.0, -178000.0,
            -122.3364486694336, infval };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        {
            float fval1 = results[idx];
            float fval2 = expected[idx];
            if( std::isnan(fval1) )
                EXPECT_TRUE( std::isnan(fval2) );
            else
                EXPECT_EQ(fval1,fval2);
        }
    }

    {    
        double nanval = std::numeric_limits<double>::quiet_NaN();
        double infval = std::numeric_limits<double>::infinity();
        thrust::device_vector<double> results(hstrs.size(),0);
        strs->stod(results.data().get());
        double expected[] = { 1234.0, 0,
            -876.0, 543.2, -0.12, 0.25, -0.002,
            0, nanval, 0, 123.0, 456.0, -178000.0,
            -122.3364478212345, infval };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        {
            double fval1 = results[idx];
            double fval2 = expected[idx];
            if( std::isnan(fval1) )
                EXPECT_TRUE( std::isnan(fval2) );
            else
                EXPECT_EQ(fval1,fval2);
        }
    }

    NVStrings::destroy(strs);
}

TEST_F(TestConvert, FromFloat)
{
    {    
        float values[] = {100, 654321.25, -12761.125, 0, 5, -4, std::numeric_limits<float>::quiet_NaN()};
        thrust::device_vector<float> results(7);
        cudaMemcpy( results.data().get(), values, 7*sizeof(float), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::ftos(results.data().get(),7);
        const char* expected[] = { "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    {    
        double values[] = {0.0000012345, 65432125000, -12761.125, 0, 5, -4, std::numeric_limits<double>::infinity()};
        thrust::device_vector<double> results(7);
        cudaMemcpy( results.data().get(), values, 7*sizeof(double), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::dtos(results.data().get(),7);
        const char* expected[] = { "1.2345e-06", "6.5432125e+10", "-12761.125", "0.0", "5.0", "-4.0", "Inf" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
}

TEST_F(TestConvert, ToBool)
{
    std::vector<const char*> hstrs{"false", nullptr, "", "true", "True", "False"};
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    thrust::device_vector<bool> results(hstrs.size(),0);
    strs->to_bools(results.data().get(), "true");
    bool expected[] = { false, false, false, true, false, false };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);
    NVStrings::destroy(strs);
}

TEST_F(TestConvert, FromBool)
{
    bool values[] = { true, false, false, true, true, true };
    thrust::device_vector<bool> results(6);
    cudaMemcpy( results.data().get(), values, 6*sizeof(bool), cudaMemcpyHostToDevice);
    NVStrings* got = NVStrings::create_from_bools(results.data().get(),6, "true", "false");
    const char* expected[] = { "true", "false", "false", "true", "true", "true" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
}

TEST_F(TestConvert, ToIPv4)
{
    std::vector<const char*> hstrs{ nullptr, "", "hello", "41.168.0.1", "127.0.0.1", "41.197.0.1" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    thrust::device_vector<unsigned int> results(hstrs.size(),0);
    strs->ip2int(results.data().get());
    unsigned int expected[] = { 0,0,0, 698875905, 2130706433, 700776449 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);
    NVStrings::destroy(strs);
}

TEST_F(TestConvert, FromIPv4)
{
    unsigned values[] = { 3232235521, 167772161, 0, 0, 700055553, 700776449 };
    thrust::device_vector<unsigned int> results(6);
    cudaMemcpy( results.data().get(), values, 6*sizeof(unsigned int), cudaMemcpyHostToDevice);
    NVStrings* got = NVStrings::int2ip(results.data().get(),6);
    const char* expected[] = { "192.168.0.1", "10.0.0.1", "0.0.0.0", "0.0.0.0", "41.186.0.1", "41.197.0.1" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
}
