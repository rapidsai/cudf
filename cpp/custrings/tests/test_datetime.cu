#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"

#include "./utils.h"

struct TestTimestamp : public GdfTest{};

TEST_F(TestTimestamp, ToTimestamp)
{
    {    
        std::vector<const char*> hstrs{"1974-02-28T01:23:45Z", "2019-07-17T21:34:37Z",
                                       nullptr, "" };
        NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
        thrust::device_vector<unsigned long> results(hstrs.size(),0);
        strs->timestamp2long("%Y-%m-%dT%H:%M:%SZ", NVStrings::seconds, results.data().get());
        int expected[] = { 131246625, 1563399277, 0,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ((int)results[idx],expected[idx]);
        NVStrings::destroy(strs);
    }

    {    
        std::vector<const char*> hstrs{"12.28.1982", "07.17.2019" };
        NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
        thrust::device_vector<unsigned long> results(hstrs.size(),0);
        strs->timestamp2long("%m-%d-%Y", NVStrings::days, results.data().get());
        int expected[] = { 4744, 18094 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ((int)results[idx],expected[idx]);
        NVStrings::destroy(strs);
    }
}

TEST_F(TestTimestamp, FromTimestamp)
{
    {    
        unsigned long values[] = {1563399273};
        thrust::device_vector<unsigned long> results(1);
        cudaMemcpy( results.data().get(), values, 1*sizeof(unsigned long), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::long2timestamp(results.data().get(),1,NVStrings::seconds,"%m/%d/%Y %H:%M");
        const char* expected[] = { "07/17/2019 21:34" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    {    
        unsigned long values[] = {1563399273123};
        thrust::device_vector<unsigned long> results(1);
        cudaMemcpy( results.data().get(), values, 1*sizeof(unsigned long), cudaMemcpyHostToDevice);
        NVStrings* got = NVStrings::long2timestamp(results.data().get(),1,NVStrings::ms,"%H:%M:%S.%f");
        const char* expected[] = { "21:34:33.123" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
}
