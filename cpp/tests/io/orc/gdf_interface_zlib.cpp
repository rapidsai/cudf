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

TEST(gdf_orc_read_zlib, testDate2038) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testDate2038.orc");

    const int num_rows = 212000;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(2, arg.num_cols_out);

#if defined(UN_COMP)
    EXPECT_EQ(std::string("col1"), arg.data[0]->col_name);
    EXPECT_EQ(std::string("col2"), arg.data[1]->col_name);
#else
    EXPECT_EQ(std::string("time"), arg.data[0]->col_name);
    EXPECT_EQ(std::string("date"), arg.data[1]->col_name);
#endif

    EXPECT_EQ(0, arg.data[0]->null_count);
    EXPECT_EQ(NULL, arg.data[0]->valid);
    EXPECT_EQ(GDF_TIMESTAMP, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);

    EXPECT_EQ(0, arg.data[1]->null_count);
    EXPECT_EQ(NULL, arg.data[1]->valid);
    EXPECT_EQ(GDF_DATE64, arg.data[1]->dtype);
    EXPECT_EQ(num_rows, arg.data[1]->size);

    orc_sint64 *ts = reinterpret_cast<orc_sint64*>(arg.data[0]->data);
    orc_sint64 *date = reinterpret_cast<orc_sint64*>(arg.data[1]->data);

    EXPECT_EQ(convertDateToGdfDate64(2038, 12, 25), date[0]);
    EXPECT_EQ(convertDateToGdfDate64(2038, 12, 25), date[999]);

    orc_sint64 gdf_ts = convertGdfTimestampMsPDT(2038, 5, 5, 12, 34, 56, 100);
    EXPECT_EQ(convertGdfTimestampMsPDT(2038, 5, 5, 12, 34, 56, 100), ts[0]);
    EXPECT_EQ(convertGdfTimestampMsPDT(2038, 5, 5, 12, 34, 56, 101), ts[13]);
    EXPECT_EQ(convertGdfTimestampMsPDT(2038, 5, 5, 12, 34, 56, 199), ts[999]);

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_zlib, testDate1900) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testDate1900.orc");

    const int num_rows = 70000;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(2, arg.num_cols_out);

    //    "type": "struct<time:timestamp,date:date>",
    EXPECT_EQ(std::string("time"), arg.data[0]->col_name);
    EXPECT_EQ(std::string("date"), arg.data[1]->col_name);

    EXPECT_EQ(0, arg.data[0]->null_count);
    EXPECT_EQ(NULL, arg.data[0]->valid);
    EXPECT_EQ(GDF_TIMESTAMP, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);

    EXPECT_EQ(0, arg.data[1]->null_count);
    EXPECT_EQ(NULL, arg.data[1]->valid);
    EXPECT_EQ(GDF_DATE64, arg.data[1]->dtype);
    EXPECT_EQ(num_rows, arg.data[1]->size);

    orc_sint64 *ts = reinterpret_cast<orc_sint64*>(arg.data[0]->data);
    orc_sint64 *date = reinterpret_cast<orc_sint64*>(arg.data[1]->data);

    //    1: {"time": "1900-05-05 12:34:56.1", "date" : "1900-12-25"}
    //    1000: {"time": "1900-05-05 12:34:56.1999", "date" : "1900-12-25"}
    //    70000: {"time": "1969-05-05 12:34:56.1999", "date": "1969-12-25"}

    EXPECT_EQ(convertDateToGdfDate64(1900, 12, 25), date[0]);
    EXPECT_EQ(convertDateToGdfDate64(1900, 12, 25), date[999]);
    EXPECT_EQ(convertDateToGdfDate64(1969, 12, 25), date[num_rows - 1]);

    // summer time was not introduced before 1918
    orc_sint64 gdf_ts = convertGdfTimestampMsPST(1900, 5, 5, 12, 34, 56, 0);
    EXPECT_EQ(convertGdfTimestampMsPST(1900, 5, 5, 12, 34, 56, 100), ts[0]);
    EXPECT_EQ(convertGdfTimestampMsPST(1900, 5, 5, 12, 34, 56, 101), ts[13]);
    EXPECT_EQ(convertGdfTimestampMsPST(1900, 5, 5, 12, 34, 56, 199), ts[999]);
    EXPECT_EQ(convertGdfTimestampMsPDT(1969, 5, 5, 12, 34, 56, 199), ts[num_rows - 1]);

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_zlib, testTimestamp) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testTimestamp.orc");
    const int num_rows = 12;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(1, arg.num_cols_out);

    EXPECT_EQ(std::string(""), arg.data[0]->col_name);

    EXPECT_EQ(0, arg.data[0]->null_count);
    EXPECT_EQ(NULL, arg.data[0]->valid);
    EXPECT_EQ(GDF_TIMESTAMP, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);

    orc_sint64 *ts = reinterpret_cast<orc_sint64*>(arg.data[0]->data);

    /*  the exact values:
        "2037-01-01 00:00:00.000999"
        "2003-01-01 00:00:00.000000222"
        "1999-01-01 00:00:00.999999999"
        "1995-01-01 00:00:00.688888888"
        "2002-01-01 00:00:00.1"
        "2010-03-02 00:00:00.000009001"
        "2005-01-01 00:00:00.000002229"
        "2006-01-01 00:00:00.900203003"
        "2003-01-01 00:00:00.800000007"
        "1996-08-02 00:00:00.723100809"
        "1998-11-02 00:00:00.857340643"
        "2008-10-02 00:00:00.0"
    */

    orc_sint64 gdf_ts = convertGdfTimestampMsPST(2037, 1, 1, 0, 0, 0, 0);
    orc_sint64 gdf_ts0 = convertGdfTimestampMsPDT(1996, 8, 2, 1);
    orc_sint64 gdf_ts1 = convertGdfTimestampMsPDT(2008, 10, 2);

    EXPECT_EQ(convertGdfTimestampMsPST(2037, 1, 1, 0, 0, 0, 0), ts[0]);
    EXPECT_EQ(convertGdfTimestampMsPST(2003, 1, 1, 0, 0, 0, 0), ts[1]);
    EXPECT_EQ(convertGdfTimestampMsPST(1999, 1, 1, 0, 0, 0, 999), ts[2]);
    EXPECT_EQ(convertGdfTimestampMsPST(1995, 1, 1, 0, 0, 0, 688), ts[3]);
    EXPECT_EQ(convertGdfTimestampMsPST(2002, 1, 1, 0, 0, 0, 100), ts[4]);
    EXPECT_EQ(convertGdfTimestampMsPST(2010, 3, 2, 0, 0, 0, 0), ts[5]);
    EXPECT_EQ(convertGdfTimestampMsPST(2005, 1, 1, 0, 0, 0, 0), ts[6]);
    EXPECT_EQ(convertGdfTimestampMsPST(2006, 1, 1, 0, 0, 0, 900), ts[7]);
    EXPECT_EQ(convertGdfTimestampMsPST(2003, 1, 1, 0, 0, 0, 800), ts[8]);
    EXPECT_EQ(convertGdfTimestampMsPDT(1996, 8, 2, 0, 0, 0, 723), ts[9]); // summer time adjustment
    EXPECT_EQ(convertGdfTimestampMsPST(1998, 11, 2, 0, 0, 0, 857), ts[10]);
    EXPECT_EQ(convertGdfTimestampMsPDT(2008, 10, 2, 0, 0, 0, 0), ts[11]); // summer time adjustment

    release_orc_read_arg(&arg);
}


TEST(gdf_orc_read_zlib, demo_11_zlib) {
    // 385 stripes
    test_demo_11_read("examples/demo-11-zlib.orc");
}

TEST(gdf_orc_read_zlib, demo_12_zlib) {
    // single stripe 
    test_demo_11_read("examples/demo-12-zlib.orc");
}

TEST(gdf_orc_read_zlib, orc_split_elim_new) {
    // single stripe
    // timezone is America/Los_Angeles (PST8)
#ifdef ORC_CONVERT_TIMESTAMP_GMT
    int zone_adjst = 8 * 3600;
#else
    int zone_adjst = 0;
#endif
    test_orc_split_elim("examples/orc_split_elim_new.orc", zone_adjst);
}

#ifdef DO_UNSUPPORTED_TEST
TEST(gdf_orc_read_zlib, orc_no_format) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/orc_no_formatp.orc");
//    release_orc_read_arg(&arg);
}
#endif

#ifdef DO_UNSUPPORTED_TEST
TEST(gdf_orc_read_zlib, over1k_bloom) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/over1k_bloom.orc");

    const int num_rows = 2098;
    
    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);

#ifdef SKIP_DECIMAL_CHECK
    const int num_cols = 10;
    const int binary_id = 9;
#else
    const int num_cols = 11;
    const int binary_id = 10;
#endif

    // "type": "struct<_col0:tinyint,_col1:smallint,_col2:int,_col3:bigint,_col4:float,_col5:double,
    //                 _col6:boolean,_col7:string,_col8:timestamp,_col9:decimal(4,2),_col10:binary>",

    EXPECT_EQ(num_cols, arg.num_cols_out);
    EXPECT_EQ(GDF_INT8,  arg.data[0]->dtype);
    EXPECT_EQ(GDF_INT16, arg.data[1]->dtype);
    EXPECT_EQ(GDF_INT32, arg.data[2]->dtype);
    EXPECT_EQ(GDF_INT64, arg.data[3]->dtype);
    EXPECT_EQ(GDF_FLOAT32, arg.data[4]->dtype);
    EXPECT_EQ(GDF_FLOAT64, arg.data[5]->dtype);
    EXPECT_EQ(GDF_INT8, arg.data[6]->dtype);    // boolean is converted into int8 so far
    EXPECT_EQ(GDF_STRING, arg.data[7]->dtype);
    EXPECT_EQ(GDF_TIMESTAMP, arg.data[8]->dtype);
    EXPECT_EQ(GDF_STRING, arg.data[binary_id]->dtype);  // binary is converted into int8 so far

    // ToDo: null bitmap check

#ifndef SKIP_DECIMAL_CHECK
    EXPECT_EQ(GDF_DECIMAL, arg.data[9]->dtype);
#endif

    for (int i = 0; i < num_cols; i++) {
        EXPECT_EQ(num_rows, arg.data[0]->size);
    }

    orc_sint8* dint8 = reinterpret_cast<orc_sint8*>(arg.data[0]->data);
    orc_sint16* dint16 = reinterpret_cast<orc_sint16*>(arg.data[1]->data);
    orc_sint32* dint32 = reinterpret_cast<orc_sint32*>(arg.data[2]->data);
    orc_sint64* dint64 = reinterpret_cast<orc_sint64*>(arg.data[3]->data);
    orc_float32* dfloat32 = reinterpret_cast<orc_float32*>(arg.data[4]->data);
    orc_float64* dfloat64 = reinterpret_cast<orc_float64*>(arg.data[5]->data);

    orc_uint8* dbool = reinterpret_cast<orc_uint8*>(arg.data[6]->data);
    orc_sint64* dts = reinterpret_cast<orc_sint64*>(arg.data[8]->data);

    gdf_string *dstring = reinterpret_cast<gdf_string*>(arg.data[7]->data);
    gdf_string *dbinary = reinterpret_cast<gdf_string*>(arg.data[binary_id]->data);


    /*  the exact values:
    0: {"_col0": 124, "_col1": 336, "_col2": 65664, "_col3": 4294967435, "_col4": 74.72, "_col5": 42.47, "_col6": true, "_col7": "bob davidson", "_col8": "2013-03-01 09:11:58.703302", "_col9": 45.40, "_col10": [1, 121, 97, 114, 100, 32, 100, 117, 116, 121, 2]}
    1105: {"_col0": 120, "_col1" : 325, "_col2" : 65758, "_col3" : 4294967540, "_col4" : 79.19, "_col5" : 11.26, "_col6" : true, "_col7" : "wendy underhill", "_col8" : "2013-03-01 09:11:58.703226", "_col9" : 94.90, "_col10" : [1, 100, 101, 98, 97, 116, 101, 2]}
    1106: {"_col0": -100, "_col1" : -1000, "_col2" : -10000, "_col3" : -1000000, "_col4" : -100, "_col5" : -10, "_col6" : false, "_col7" : null, "_col8" : null, "_col9" : null, "_col10" : null}
    */

    int id = 0;
    EXPECT_EQ(dint8[id], orc_sint8(124));
    EXPECT_EQ(dint16[id], orc_sint16(336));
    EXPECT_EQ(dint32[id], orc_sint32(65664));
    EXPECT_EQ(dint64[id], orc_sint64(4294967435));
    EXPECT_EQ(dfloat32[id], 74.72f);
    EXPECT_EQ(dfloat64[id], 42.47);
    EXPECT_EQ(dbool[id], 1);

#if 0   // bug
    EXPECT_EQ_STR(dstring[id], gdf_string("bob davidson", strlen("bob davidson")));
    EXPECT_EQ_STR(dbinary[id], gdf_string("\1\121\97\114\100\32\100\117\116\121\2", 11));
#endif

    id = 1048;
    EXPECT_EQ(dint8[id], orc_sint8(120));
    EXPECT_EQ(dint16[id], orc_sint16(325));
    EXPECT_EQ(dint32[id], orc_sint32(65758));
    EXPECT_EQ(dint64[id], orc_sint64(4294967540));
    EXPECT_EQ(dfloat32[id], 79.19f);
    EXPECT_EQ(dfloat64[id], 11.26);
    EXPECT_EQ(dbool[id], 1);
 
#if 0
    EXPECT_EQ_STR(dstring[id], gdf_string("wendy underhill", strlen("wendy underhill")));
    EXPECT_EQ_STR(dbinary[id], gdf_string("\1\100\101\98\97\116\101\2", 8));
#endif

    EXPECT_EQ(convertGdfTimestampMs(2013, 3, 1, 9, 11, 58, 703), dts[0]);
    EXPECT_EQ(convertGdfTimestampMs(2013, 3, 1, 9, 11, 58, 703), dts[1105]);

    for (id = 1049; id < 2098; id++) {
        EXPECT_EQ(dint8[id], orc_sint8(-100));
        EXPECT_EQ(dint16[id], orc_sint16(-1000));
        EXPECT_EQ(dint32[id], orc_sint32(-10000));
        EXPECT_EQ(dint64[id], orc_sint64(-1000000));
        EXPECT_EQ(dfloat32[id], -100.0f);
        EXPECT_EQ(dfloat64[id], -10.0);
        EXPECT_EQ(dbool[id], 0);
    }
    
    release_orc_read_arg(&arg);
}
#endif

TEST(gdf_orc_read_zlib, orc_index_int_string) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/orc_index_int_string.orc");
    const int num_rows = 6000;

    ASSERT_EQ(GDF_SUCCESS, ret);  // int column is loaded, but string column is skipped for now.
    EXPECT_EQ(num_rows, arg.num_rows_out);

    EXPECT_EQ(2, arg.num_cols_out);
    EXPECT_EQ(GDF_INT32, arg.data[0]->dtype);
    EXPECT_EQ(num_rows, arg.data[0]->size);

    int *expected_int = new int[num_rows];
    for (int i = 0; i < num_rows; i++)expected_int[i] = i + 1;

    compare_arrays(expected_int, reinterpret_cast<int*>(arg.data[0]->data), num_rows);

    EXPECT_EQ(GDF_STRING, arg.data[1]->dtype);
    EXPECT_EQ(num_rows, arg.data[1]->size);

    gdf_string *dstr1 = reinterpret_cast<gdf_string*>(arg.data[1]->data);

#if defined(DO_FULL_RANGE_CHECK)
    for (int i = 0; i < 999; i++) {
        char buf[32];
        sprintf(buf, "%da", i + 1);
        EXPECT_EQ_STR(dstr1[i], gdf_string(buf, strlen(buf)));
    }

    for (int i = 1000; i < 6000; i++) {
        char buf[32];
        sprintf(buf, "%d", i + 1);
        EXPECT_EQ_STR(dstr1[i], gdf_string(buf, strlen(buf)));
    }
#endif

    release_orc_read_arg(&arg);
}

#ifdef DO_UNSUPPORTED_TEST
TEST(gdf_orc_read_zlib, test1) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.test1.orc");
    const int num_rows = 2;

#ifdef SKIP_LIST_MAP
    const int num_cols = 9;
#else
    const int num_cols = 9;
#endif

    EXPECT_EQ(num_rows, arg.num_rows_out);
    EXPECT_EQ(num_cols, arg.num_cols_out);

    // "type": "struct<boolean1:boolean,byte1:tinyint,short1:smallint,int1:int,long1:bigint,
    // float1:float,double1:double,bytes1:binary,string1:string,
    // middle:struct<list:array<struct<int1:int,string1:string>>>,
    // list:array<struct<int1:int,string1:string>>,map:map<string,struct<int1:int,string1:string>>>",

    EXPECT_EQ(GDF_INT8, arg.data[0]->dtype);    // boolean is converted into int8 so far
    EXPECT_EQ(GDF_INT8, arg.data[1]->dtype);
    EXPECT_EQ(GDF_INT16, arg.data[2]->dtype);
    EXPECT_EQ(GDF_INT32, arg.data[3]->dtype);
    EXPECT_EQ(GDF_INT64, arg.data[4]->dtype);
    EXPECT_EQ(GDF_FLOAT32, arg.data[5]->dtype);
    EXPECT_EQ(GDF_FLOAT64, arg.data[6]->dtype);
    EXPECT_EQ(GDF_STRING, arg.data[7]->dtype);  // binary is converted into int8 so far
    EXPECT_EQ(GDF_STRING, arg.data[8]->dtype);

    const char* col_names[] = { "boolean1", "byte1", "short1", "int1", "long1",
        "float1", "double1", "bytes1", "string1"
    };

    for (int i = 0; i < num_cols; i++) {
        EXPECT_EQ(std::string(col_names[i]), arg.data[i]->col_name);
    }

    orc_sint8* dbool = reinterpret_cast<orc_sint8*>(arg.data[0]->data);
    orc_sint8* dint8 = reinterpret_cast<orc_sint8*>(arg.data[1]->data);
    orc_sint16* dint16 = reinterpret_cast<orc_sint16*>(arg.data[2]->data);
    orc_sint32* dint32 = reinterpret_cast<orc_sint32*>(arg.data[3]->data);
    orc_sint64* dint64 = reinterpret_cast<orc_sint64*>(arg.data[4]->data);
    orc_float32* dfloat32 = reinterpret_cast<orc_float32*>(arg.data[5]->data);
    orc_float64* dfloat64 = reinterpret_cast<orc_float64*>(arg.data[6]->data);

    gdf_string *dbinary = reinterpret_cast<gdf_string*>(arg.data[7]->data);
    gdf_string *dstring = reinterpret_cast<gdf_string*>(arg.data[8]->data);

    /* the exact values
    {"boolean1": false, "byte1" : 1, "short1" : 1024, "int1" : 65536, "long1" : 9223372036854775807, 
        "float1" : 1, "double1" : -15, "bytes1" : [0, 1, 2, 3, 4], "string1" : "hi", 
        "middle" : {"list": [{"int1": 1, "string1" : "bye"}, { "int1": 2, "string1" : "sigh" }]}, 
        "list" : [{"int1": 3, "string1" : "good"}, { "int1": 4, "string1" : "bad" }], "map" : []}
    {"boolean1": true, "byte1" : 100, "short1" : 2048, "int1" : 65536, "long1" : 9223372036854775807,
        "float1" : 2, "double1" : -5, "bytes1" : [], "string1" : "bye",
        "middle" : {"list": [{"int1": 1, "string1" : "bye"}, { "int1": 2, "string1" : "sigh" }]},
        "list" : [{"int1": 100000000, "string1" : "cat"}, { "int1": -100000, "string1" : "in" }, { "int1": 1234, "string1" : "hat" }],
        "map" : [{"key": "chani", "value" : {"int1": 5, "string1" : "chani"}}, { "key": "mauddib", "value" : {"int1": 1, "string1" : "mauddib"} }]}
    */

    int id = 0;
    EXPECT_EQ(dbool[id], 0);
    EXPECT_EQ(dint8[id], orc_sint8(1));
    EXPECT_EQ(dint16[id], orc_sint16(1024));
    EXPECT_EQ(dint32[id], orc_sint32(65536));
    EXPECT_EQ(dint64[id], orc_sint64(9223372036854775807));
    EXPECT_EQ(dfloat32[id], 1.f);
    EXPECT_EQ(dfloat64[id], -15.0);
    EXPECT_EQ_STR(dbinary[id], gdf_string("\0\1\2\3\4", 5));
    EXPECT_EQ_STR(dstring[id], gdf_string("hi", strlen("hi")));

    id = 1;
    EXPECT_EQ(dbool[id], 1);
    EXPECT_EQ(dint8[id], orc_sint8(100));
    EXPECT_EQ(dint16[id], orc_sint16(2048));
    EXPECT_EQ(dint32[id], orc_sint32(65536));
    EXPECT_EQ(dint64[id], orc_sint64(9223372036854775807));
    EXPECT_EQ(dfloat32[id], 2.f);
    EXPECT_EQ(dfloat64[id], -5.0);
    EXPECT_EQ_STR(dbinary[id], gdf_string("", 0));
    EXPECT_EQ_STR(dstring[id], gdf_string("bye", strlen("bye")));


    release_orc_read_arg(&arg);
}
#endif

#ifdef DO_UNSUPPORTED_TEST
TEST(gdf_orc_read_zlib, testSeek) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testSeek.orc");
    const int num_rows = 32768;

    /* using list, map, byte, binary
    "type": "struct<boolean1:boolean,byte1:tinyint,short1:smallint,int1:int,long1:bigint,
        float1:float,double1:double,bytes1:binary,string1:string,
        middle:struct<list:array<struct<int1:int,string1:string>>>,
        list:array<struct<int1:int,string1:string>>,
        map:map<string,struct<int1:int,string1:string>>>",
    */

    release_orc_read_arg(&arg);
}
#endif


TEST(gdf_orc_read_zlib, testStringAndBinaryStatistics) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testStringAndBinaryStatistics.orc");
    const int num_rows = 4;

    // "type": "struct<bytes1:binary,string1:string>",
    ASSERT_EQ(GDF_SUCCESS, ret);
    EXPECT_EQ(num_rows, arg.num_rows_out);

    // for now, binary is treated as string data type
    EXPECT_EQ(2, arg.num_cols_out);
    EXPECT_EQ(GDF_STRING, arg.data[0]->dtype);  
    EXPECT_EQ(GDF_STRING, arg.data[1]->dtype);

    EXPECT_EQ(num_rows, arg.data[0]->size);
    EXPECT_EQ(num_rows, arg.data[1]->size);

    EXPECT_EQ(std::string("bytes1"), arg.data[0]->col_name);
    EXPECT_EQ(std::string("string1"), arg.data[1]->col_name);

    // check data has null bitmap
    EXPECT_NE(nullptr, arg.data[0]->valid);
    EXPECT_NE(nullptr, arg.data[1]->valid);

    // check the null bitmap values
    EXPECT_EQ(0x07, arg.data[0]->valid[0]);
    EXPECT_EQ(0x0b, arg.data[1]->valid[0]);

    /*  the exact values:
    {"bytes1": [0, 1, 2, 3, 4], "string1": "foo"}
    {"bytes1": [0, 1, 2, 3], "string1": "bar"}
    {"bytes1": [0, 1, 2, 3, 4, 5], "string1": null}
    {"bytes1": null, "string1": "hi"}
    */

    const char* byteArray[] = {
        "\0\1\2\3\4", "\0\1\2\3", "\0\1\2\3\4\5", NULL
    };
    int sizeof_byteArray[] = { 5, 4, 6, 0 };
    const char* stringArray[] = {
        "foo", "bar", NULL, "hi"
    };
    int sizeof_stringArray[] = { 3, 3, 0, 2 };

    gdf_string *bytes1 = reinterpret_cast<gdf_string*>(arg.data[0]->data);
    gdf_string *string1 = reinterpret_cast<gdf_string*>(arg.data[1]->data);

    gdf_string sff(stringArray[0], strlen(stringArray[0]));

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ_STR(bytes1[i],  gdf_string(byteArray[i], sizeof_byteArray[i]));
        EXPECT_EQ_STR(string1[i], gdf_string(stringArray[i], sizeof_stringArray[i]));
    }

    release_orc_read_arg(&arg);
}


TEST(gdf_orc_read_zlib, testStripeLevelStats) {
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/TestOrcFile.testStripeLevelStats.orc");
    const int num_rows = 11000;
    ASSERT_EQ(GDF_SUCCESS, ret);

    // struct<int1:int,string1:string>,
    // for now, binary is treated as string data type
    EXPECT_EQ(2, arg.num_cols_out);
    EXPECT_EQ(GDF_INT32,  arg.data[0]->dtype);
    EXPECT_EQ(GDF_STRING, arg.data[1]->dtype);

    EXPECT_EQ(num_rows, arg.data[0]->size);
    EXPECT_EQ(num_rows, arg.data[1]->size);

    EXPECT_EQ(std::string("int1"), arg.data[0]->col_name);
    EXPECT_EQ(std::string("string1"), arg.data[1]->col_name);

    orc_sint32 *dint = reinterpret_cast<orc_sint32*>(arg.data[0]->data);
    gdf_string *dstr = reinterpret_cast<gdf_string*>(arg.data[1]->data);

    for (int i = 0; i < 5000; i++) {
        EXPECT_EQ(dint[i], 1);
        EXPECT_EQ_STR(dstr[i], gdf_string("one", 3));
    }

    for (int i = 5000; i < 10000; i++) {
        EXPECT_EQ(dint[i], 2);
        EXPECT_EQ_STR(dstr[i], gdf_string("two", 3));
    }

    for (int i = 10000; i < 11000; i++) {
        EXPECT_EQ(dint[i], 3);
        EXPECT_EQ_STR(dstr[i], gdf_string("three", 5));
    }
    release_orc_read_arg(&arg);
}

#endif // #ifndef GDF_ORC_NO_FILE_TEST