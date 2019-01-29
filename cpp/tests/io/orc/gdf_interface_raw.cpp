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

// unit_test.cpp : Defines the entry point for the console application.
//

#include "gdf_interface.h"


#ifndef GDF_ORC_NO_FILE_TEST

#define CUDA_TRY_FREE(val) if(val){CudaFuncCall(cudaFree(val));};

gdf_error release_orc_read_arg(orc_read_arg* arg)
{
    if (!arg->data)return GDF_SUCCESS;

    for (int i = 0; i < arg->num_cols_out; i++) {
        if (arg->data && arg->data[i]) {
            gdf_orc_release_column_name(arg->data[i]);
#ifdef HAS_GDF_LIB
            gdf_column_free(arg->data[i]);
#else
            CUDA_TRY_FREE(arg->data[i]->data);
            CUDA_TRY_FREE(arg->data[i]->valid);
#endif
        }
    }

    free(arg->data);
    arg->data = NULL;
    return GDF_SUCCESS;
}

void test_orc_split_elim(const char* filename, int adjsec)
{
    // orc_split_elim.orc is fully coverted.
    // it has int, string, double, timestamp columns over multi stripes
    // this file never contain any null bitmps

    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, filename);
    const int num_rows = 25000;

    ASSERT_EQ(GDF_SUCCESS, ret);   // check the error code
    EXPECT_EQ(4, arg.num_cols_out);   // this is 3 for now, but 5 in future if the reader support string.

    int id = 0;
    EXPECT_EQ(GDF_INT64, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(std::string("userid"), arg.data[id]->col_name);

    id = 1;
    EXPECT_EQ(GDF_STRING, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(std::string("string1"), arg.data[id]->col_name);

    id = 2;
    EXPECT_EQ(GDF_FLOAT64, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(std::string("subtype"), arg.data[id]->col_name);

    id = 3;
    EXPECT_EQ(GDF_TIMESTAMP, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(std::string("ts"), arg.data[id]->col_name);

#ifndef IGNORE_NULL_BITMAP_CHECK
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(nullptr, arg.data[id]->valid);
        }
    }
#endif

    /*  the reference values:
    0: {"userid": 2, "string1" : "foo", "subtype" : 0.8, "decimal1" : 1.200000, "ts" : "1969-12-31 16:00:00.0"}
    1: {"userid": 100, "string1" : "zebra", "subtype" : 8, "decimal1" : 0.000000, "ts" : "1969-12-31 16:04:10.0"}
    5000: {"userid": 13, "string1" : "bar", "subtype" : 80, "decimal1" : 2.200000, "ts" : "1969-12-31 16:00:05.0"}
    10000: {"userid": 29, "string1" : "cat", "subtype" : 8, "decimal1" : 3.300000, "ts" : "1969-12-31 16:00:10.0"}
    15000: {"userid": 70, "string1" : "dog", "subtype" : 1.8, "decimal1" : 4.400000, "ts" : "1969-12-31 16:00:15.0"}
    20000: {"userid": 5, "string1" : "eat", "subtype" : 0.8, "decimal1" : 5.500000, "ts" : "1969-12-31 16:00:20.0"}
    */

    orc_sint64 *dlong1 = reinterpret_cast<orc_sint64*>(arg.data[0]->data);
    gdf_string *dstring = reinterpret_cast<gdf_string*>(arg.data[1]->data);
    double *ddouble1 = reinterpret_cast<double*>(arg.data[2]->data);
    orc_sint64 *dts = reinterpret_cast<orc_sint64*>(arg.data[3]->data);

    EXPECT_EQ(dlong1[0], 2);
    EXPECT_EQ(dlong1[1], 100);
    EXPECT_EQ(dlong1[5000], 13);
    EXPECT_EQ(dlong1[10000], 29);
    EXPECT_EQ(dlong1[15000], 70);
    EXPECT_EQ(dlong1[20000], 5);
    EXPECT_EQ(dlong1[num_rows - 1], 100);

    EXPECT_EQ(ddouble1[0], 0.8);
    EXPECT_EQ(ddouble1[1], 8.0);
    EXPECT_EQ(ddouble1[5000], 80.0);
    EXPECT_EQ(ddouble1[10000], 8.0);
    EXPECT_EQ(ddouble1[15000], 1.8);
    EXPECT_EQ(ddouble1[20000], 0.8);
    EXPECT_EQ(ddouble1[num_rows - 1], 8.0);

    // this is encoded by dictionary mode.
    EXPECT_EQ_STR(dstring[0], gdf_string("foo", 3));
    EXPECT_EQ_STR(dstring[1], gdf_string("zebra", 5));
    EXPECT_EQ_STR(dstring[5000], gdf_string("bar", 3));
    EXPECT_EQ_STR(dstring[10000], gdf_string("cat", 3));
    EXPECT_EQ_STR(dstring[15000], gdf_string("dog", 3));
    EXPECT_EQ_STR(dstring[20000], gdf_string("eat", 3));
    EXPECT_EQ_STR(dstring[num_rows - 1], gdf_string("zebra", 5));

#define convertGdfTimestampMsAdj(y, m, d, h, min, sec) convertGdfTimestampMs(y, m, d, h, min, sec, 0, adjsec)

    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 0, 0), dts[0]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 4, 10), dts[1]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 0, 5), dts[5000]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 0, 10), dts[10000]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 0, 15), dts[15000]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 0, 20), dts[20000]);
    EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 4, 10), dts[num_rows - 1]);

#if defined(DO_FULL_RANGE_CHECK)
    for (int i = 2; i < num_rows - 1; i++) {
        if (i % 5000 == 0)continue;
        EXPECT_EQ(dlong1[i], 100);
        EXPECT_EQ(ddouble1[i], 8);
        EXPECT_EQ_STR(dstring[i], gdf_string("zebra", 5));
        EXPECT_EQ(convertGdfTimestampMsAdj(1969, 12, 31, 16, 4, 10), dts[i]);
    }
#endif
#undef convertGdfTimestampMsAdj

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_raw, orc_split_elim)
{
    // 5 stripes
    // this file does not need to support null bitmap, however it has null bitmap for now
    // ToDo: further null bitmap check
    test_orc_split_elim("examples/orc_split_elim.orc");
}

TEST(gdf_orc_read_raw, columnProjection)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.columnProjection.orc");
    const int num_rows = 21000;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);

    EXPECT_EQ(2, arg.num_cols_out);
    EXPECT_EQ(GDF_INT32, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);
    EXPECT_EQ(0, arg.data[0]->null_count);
    EXPECT_EQ(NULL, arg.data[0]->valid);
    EXPECT_EQ(std::string("int1"), arg.data[0]->col_name);

    EXPECT_EQ(GDF_STRING, arg.data[1]->dtype);
    EXPECT_EQ(num_rows, arg.data[1]->size);
    EXPECT_EQ(0, arg.data[1]->null_count);
    EXPECT_EQ(NULL, arg.data[1]->valid);
    EXPECT_EQ(std::string("string1"), arg.data[1]->col_name);

    const orc_sint32 *dint1 = reinterpret_cast<orc_sint32*>(arg.data[0]->data);
    const gdf_string *dstr1 = reinterpret_cast<gdf_string*>(arg.data[1]->data);

    // validate the part of output
    EXPECT_EQ(dint1[0], -1155869325);
    EXPECT_EQ(dint1[1], 431529176);
    EXPECT_EQ(dint1[10000], -578147096);
    EXPECT_EQ(dint1[num_rows - 1], -837225228);

    EXPECT_EQ_STR(dstr1[0], gdf_string("bb2c72394b1ab9f8", 16));
    EXPECT_EQ_STR(dstr1[1], gdf_string("e6c5459001105f17", 16));
    EXPECT_EQ_STR(dstr1[9999], gdf_string("4f50cfd1c5118898", 16));
    EXPECT_EQ_STR(dstr1[10000], gdf_string("24e956eb058bc491", 16));
    EXPECT_EQ_STR(dstr1[num_rows - 1], gdf_string("e222dcce8670f10e", 16));

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_raw, testPredicatePushdown)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testPredicatePushdown.orc");
    const int num_rows = 3500;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(2, arg.num_cols_out); // struct, int, string

    int id = 0;
    EXPECT_EQ(GDF_INT32, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(NULL, arg.data[id]->valid);
    EXPECT_EQ(std::string("int1"), arg.data[id]->col_name);

    id = 1;
    EXPECT_EQ(GDF_STRING, arg.data[id]->dtype);
    EXPECT_EQ(num_rows, arg.data[id]->size);
    EXPECT_EQ(0, arg.data[id]->null_count);
    EXPECT_EQ(NULL, arg.data[id]->valid);
    EXPECT_EQ(std::string("string1"), arg.data[id]->col_name);

    orc_sint32 *dint1 = reinterpret_cast<orc_sint32*>(arg.data[0]->data);
    gdf_string *dstring = reinterpret_cast<gdf_string*>(arg.data[1]->data);

    // full coverage
    for (int i = 0; i < num_rows; i++) {
        EXPECT_EQ(300 * i, dint1[i]);
    }

    // limited coverage, the strigs are encoded by direct V2
    EXPECT_EQ_STR(dstring[0], gdf_string("0", 1));
    EXPECT_EQ_STR(dstring[1], gdf_string("a", 1));
    EXPECT_EQ_STR(dstring[2], gdf_string("14", 2));
    EXPECT_EQ_STR(dstring[26], gdf_string("104", 3));
    EXPECT_EQ_STR(dstring[num_rows - 1], gdf_string("88ae", 4));

    release_orc_read_arg(&arg);
}



void test_demo_11_read(const char* filename) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, filename);
    const int num_rows = 1920800;    // 1.9M

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(9, arg.num_cols_out); // struct, int, string

    gdf_dtype types[] = { GDF_INT32, GDF_STRING, GDF_STRING, GDF_STRING, GDF_INT32, GDF_STRING, GDF_INT32, GDF_INT32, GDF_INT32 };
    const char* names[] = { "_col0", "_col1", "_col2", "_col3", "_col4", "_col5", "_col6", "_col7", "_col8" };

    for (int id = 0; id < 9; id++) {
        EXPECT_EQ(types[id], arg.data[id]->dtype);
        EXPECT_EQ(std::string(names[id]), arg.data[id]->col_name);

        EXPECT_EQ(num_rows, arg.data[id]->size);
#ifndef IGNORE_NULL_BITMAP_CHECK
        EXPECT_EQ(0, arg.data[id]->null_count);
        EXPECT_EQ(nullptr, arg.data[id]->valid);
#endif
    }

    orc_sint32 *dint0 = reinterpret_cast<orc_sint32*>(arg.data[0]->data);
    gdf_string *dstr1 = reinterpret_cast<gdf_string*>(arg.data[1]->data);
    gdf_string *dstr2 = reinterpret_cast<gdf_string*>(arg.data[2]->data);
    gdf_string *dstr3 = reinterpret_cast<gdf_string*>(arg.data[3]->data);
    orc_sint32 *dint4 = reinterpret_cast<orc_sint32*>(arg.data[4]->data);
    gdf_string *dstr5 = reinterpret_cast<gdf_string*>(arg.data[5]->data);
    orc_sint32 *dint6 = reinterpret_cast<orc_sint32*>(arg.data[6]->data);
    orc_sint32 *dint7 = reinterpret_cast<orc_sint32*>(arg.data[7]->data);
    orc_sint32 *dint8 = reinterpret_cast<orc_sint32*>(arg.data[8]->data);

    /*  the reference:
    0: {"_col0": 1, "_col1": "M", "_col2": "M", "_col3": "Primary", "_col4": 500, "_col5": "Good", "_col6": 0, "_col7": 0, "_col8": 0}
    1: {"_col0": 2, "_col1": "F", "_col2": "M", "_col3": "Primary", "_col4": 500, "_col5": "Good", "_col6": 0, "_col7": 0, "_col8": 0}
    30: {"_col0": 31, "_col1": "M", "_col2": "M", "_col3": "2 yr Degree", "_col4": 500, "_col5": "Good", "_col6": 0, "_col7": 0, "_col8": 0}
    1920799: {"_col0": 1920800, "_col1": "F", "_col2": "U", "_col3": "Unknown", "_col4": 10000, "_col5": "Unknown", "_col6": 6, "_col7": 6, "_col8": 6}
    */
    int last = num_rows - 1;

    EXPECT_EQ(1, dint0[0]);
    EXPECT_EQ(2, dint0[1]);
    EXPECT_EQ(31, dint0[30]);
    EXPECT_EQ(1920800, dint0[last]);

    EXPECT_EQ_STR(dstr1[0], gdf_string("M", 1));
    EXPECT_EQ_STR(dstr1[1], gdf_string("F", 1));
    EXPECT_EQ_STR(dstr1[30], gdf_string("M", 1));
    EXPECT_EQ_STR(dstr1[last], gdf_string("F", 1));

    EXPECT_EQ_STR(dstr2[0], gdf_string("M", 1));
    EXPECT_EQ_STR(dstr2[1], gdf_string("M", 1));
    EXPECT_EQ_STR(dstr2[30], gdf_string("M", 1));
    EXPECT_EQ_STR(dstr2[last], gdf_string("U", 1));

    EXPECT_EQ_STR(dstr3[0], gdf_string("Primary", 7));
    EXPECT_EQ_STR(dstr3[1], gdf_string("Primary", 7));
    EXPECT_EQ_STR(dstr3[30], gdf_string("2 yr Degree", 11));
    EXPECT_EQ_STR(dstr3[last], gdf_string("Unknown", 7));

    EXPECT_EQ(500, dint4[0]);
    EXPECT_EQ(500, dint4[1]);
    EXPECT_EQ(500, dint4[30]);
    EXPECT_EQ(10000, dint4[last]);

    EXPECT_EQ_STR(dstr5[0], gdf_string("Good", 4));
    EXPECT_EQ_STR(dstr5[1], gdf_string("Good", 4));
    EXPECT_EQ_STR(dstr5[30], gdf_string("Good", 4));
    EXPECT_EQ_STR(dstr5[last], gdf_string("Unknown", 7));

    EXPECT_EQ(0, dint6[0]);
    EXPECT_EQ(0, dint6[1]);
    EXPECT_EQ(0, dint6[30]);
    EXPECT_EQ(6, dint6[last]);

    EXPECT_EQ(0, dint7[0]);
    EXPECT_EQ(0, dint7[1]);
    EXPECT_EQ(0, dint7[30]);
    EXPECT_EQ(6, dint7[last]);

    EXPECT_EQ(0, dint8[0]);
    EXPECT_EQ(0, dint8[1]);
    EXPECT_EQ(0, dint8[30]);
    EXPECT_EQ(6, dint8[last]);

    // full range test for _col0
#if defined(DO_FULL_RANGE_CHECK)
    for (int i = 0; i < num_rows; i++) {
        EXPECT_EQ(i + 1, dint0[i]);
    }
#endif

    release_orc_read_arg(&arg);
}


TEST(gdf_orc_read_raw, demo_11_none)
{
    test_demo_11_read("examples/demo-11-none.orc");
}


#ifdef DO_UNSUPPORTED_TEST // decimal is not supported yet
TEST(gdf_orc_read_raw, decimal) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/decimal.orc");
    const int num_rows = 6000;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.

#ifdef SKIP_DECIMAL_CHECK
    EXPECT_EQ(0, arg.num_rows_out);
    EXPECT_EQ(0, arg.num_cols_out);
#else
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(1, arg.num_cols_out);
    EXPECT_EQ(GDF_INT64, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);
#endif

    // "type": "struct<_col0:decimal(10,5)>"

    //    EXPECT_EQ(0, arg.data[0]->null_count);
    //    EXPECT_EQ(NULL, arg.data[0]->valid);
    //    EXPECT_EQ(std::string("int1"), arg.data[0]->col_name);

    release_orc_read_arg(&arg);
}
#endif

#ifdef DO_UNSUPPORTED_TEST  // not supported yet
TEST(gdf_orc_read_raw, testMemoryManagementV11)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testMemoryManagementV11.orc");
    const int num_rows = 2500;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);

    EXPECT_EQ(2, arg.num_cols_out);   // this is 1 for now, but 2 in future if the reader support string.
    EXPECT_EQ(GDF_INT32, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);

#ifndef IGNORE_NULL_BITMAP_CHECK
    EXPECT_EQ(0, arg.data[0]->null_count);
    EXPECT_EQ(NULL, arg.data[0]->valid);
    EXPECT_EQ(0, arg.data[1]->null_count);
    EXPECT_EQ(NULL, arg.data[1]->valid);
#endif

    int *expected_int = new int[num_rows];
    for (int i = 0; i < num_rows; i++)expected_int[i] = 300 * i;

    compare_arrays(expected_int, reinterpret_cast<int*>(arg.data[0]->data), num_rows);

    gdf_string* dstr = reinterpret_cast<gdf_string*>(arg.data[1]->data);

#ifdef DO_FULL_RANGE_CHECK 
    for (int i = 0; i < num_rows; i++) {
        int val = i * 10;
        char buf[10];
        sprintf(buf, "%x", val);
        EXPECT_EQ_STR(dstr[i], gdf_string(buf, strlen(buf)));
    }
#endif

    delete expected_int;
    release_orc_read_arg(&arg);
}

#endif

#endif // #ifndef GDF_ORC_NO_FILE_TEST