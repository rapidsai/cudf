#include <cstdio>
#include <vector>

#include <rmm/rmm.h>

#include "nvstrings/NVStrings.h"

#define ASSERT_RMM_SUCCEEDED(expr)  ASSERT_EQ(RMM_SUCCESS, expr)

// Base class fixture for GDF google tests that initializes / finalizes the
// RAPIDS memory manager
struct GdfTest : public ::testing::Test
{
    static void SetUpTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmInitialize(nullptr) );
    }

    static void TearDownTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmFinalize() );
    }
};

// utility to verify strings results

bool verify_strings( NVStrings* d_strs, const char** h_strs )
{
    unsigned int count = d_strs->size();
    std::vector<int> bytes(count);
    d_strs->byte_count(bytes.data(),false);
    std::vector<char*> ptrs(count,nullptr);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        int size = bytes[idx];
        if( size < 0 )
            continue;
        char* str = (char*)malloc(size+1);
        str[size] = 0;
        ptrs[idx] = str;
    }
    d_strs->to_host( ptrs.data(), 0, (int)count );
    bool bmatched = true;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        char* str1 = ptrs[idx];
        const char* str2 = h_strs[idx];
        if( str1 )
        {
            if( !str2 || (strcmp(str1,str2)!=0) )
            {
                printf("%d:[%s]!=[%s]\n",idx,str1,str2);
                bmatched = false;
            }
            free(str1);
        }
        else if( str2 )
        {
            printf("%d:[%s]!=[%s]\n",idx,str1,str2);
            bmatched = false;
        }
    }
    return bmatched;
}

