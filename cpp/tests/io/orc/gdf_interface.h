/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __GDF_INTERFACE__HEADER__
#define __GDF_INTERFACE__HEADER__

#include "tests_common.h"

#include "cudf.h"
#include "io/orc/gdf_orc_util.h"
#include "io/orc/kernel_util.cuh"

#define DO_FULL_RANGE_CHECK 
#define IGNORE_NULL_BITMAP_CHECK

#if (0 || defined(ORC_DEVELOP_MODE) ) // these are debug/test flags
#define DO_UNSUPPORTED_COMP_TEST
#define DO_UNSUPPORTED_TEST
#endif

#ifdef DO_UNSUPPORTED_TEST
#define SKIP_DECIMAL_CHECK
#define SKIP_LIST_MAP
#define SKIP_UNION
#endif

inline
bool operator== (const gdf_string &c1, const gdf_string &c2)
{
    if (c1.second != c2.second)return false;
    for (int i = 0; i < c1.second; i++) {
        if (c1.first[i] != c2.first[i])return false;
    }

    return true;
}

#define EXPECT_EQ_STR(c1, c2) EXPECT_EQ(true, (c1 == c2))

gdf_error release_orc_read_arg(orc_read_arg* arg);

void test_demo_11_read(const char* filename);
void test_orc_split_elim(const char* filename, int adjsec=0);

inline gdf_error load_orc(orc_read_arg* arg, const char* filename)
{
    arg->file_path = filename;
#ifdef ORC_CONVERT_TIMESTAMP_GMT    // this should be defined at "tests_common.h" 
    arg->convertToGMT = true;
#else
    arg->convertToGMT = false;
#endif
    gdf_error ret = gdf_read_orc(arg);

    return ret;
}


#endif // __GDF_INTERFACE__HEADER__


