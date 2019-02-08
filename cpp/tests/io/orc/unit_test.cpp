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

#include "tests_common.h"

#include "io/orc/gdf_orc_util.h"
#include "io/orc/orc_read_impl_proto.h"
#include "io/orc/kernel_util.cuh"

TEST(OrcErrorCode, ConversionCoverage)
{
    // CudaOrcError_t -> gdf_error conversion coverage test
    for (int i = 0; i < GDF_ORC_MAX_ERROR_CODE; i++) {
        EXPECT_NE(-1, (int)(gdf_orc_convertErrorCode(CudaOrcError_t(i))));
    }
}

void regionCheck(int exactOffset, const char* region)
{
    int gmtoffset = 0;
    findGMToffsetFromRegion(gmtoffset, region);
    EXPECT_EQ(exactOffset, gmtoffset);
};

TEST(Utility, TimezoneFinding)
{
    regionCheck(8 * 3600, "US/Pacific");
    regionCheck(-9 * 3600, "Japan");
    regionCheck(-10 * 3600, "Australia/Sydney");
}


#ifndef GDF_ORC_NO_FILE_TEST

TEST(orc_reader_inspection, UncomressedLoad) {
    const char* filename = "examples/decimal.orc";

    CudaOrcReaderImplProto* reader = (CudaOrcReaderImplProto*)gdf_create_orc_reader();

    ASSERT_EQ(GDF_ORC_SUCCESS, reader->ReadFromFile(filename));
    const auto ft = reader->getFooter();

    // -- validate the postscript/footer info
    EXPECT_EQ(89, ft->footerLength);
    EXPECT_EQ(OrcCompressionKind::NONE, ft->compressionKind);
    EXPECT_EQ(0, ft->compressionBlockSize);
    EXPECT_EQ(44, ft->metadataLength);
    
    EXPECT_EQ(3, ft->headerlength);
    EXPECT_EQ(16186, ft->contentlength);
    EXPECT_EQ(6000, ft->numberofrows);
    EXPECT_EQ(10000, ft->rowindexstride);
    EXPECT_EQ(0, ft->user_metadata_size);
    EXPECT_EQ(2, ft->statistics_size);
    EXPECT_EQ(2, ft->types_size);
    EXPECT_EQ(1, ft->stripes_size);

    // -- validate boolean BLE
    const int block_size = 2000/8;
    orc_uint8 orc_true = 0xff;

    orc_uint8 orc_false = 0;
    orc_byte* bool_exact = (orc_byte*)malloc(block_size * 3);
    memset(bool_exact + block_size *0, orc_true,  block_size);
    memset(bool_exact + block_size *1, orc_false, block_size);
    memset(bool_exact + block_size *2, orc_true,  block_size);

    EXPECT_EQ(-1,  compare_arrays(bool_exact, (orc_byte*)(reader->getHost(1).stream_present), block_size*3) );

    orc_sint64* precision = reinterpret_cast<orc_sint64*>( reader->getHost(1).stream_data );
    orc_uint32* scale     = reinterpret_cast<orc_uint32*>( reader->getHost(1).stream_secondary );

//    EXPECT_EQ();

    EXPECT_EQ(-10005, precision[0]);
    EXPECT_EQ(1, scale[0]);
    EXPECT_EQ(9992004, precision[1999]);
    EXPECT_EQ(4, scale[1999]);

    // these values are undefined.
//    EXPECT_EQ(0, precision[2000]);
//    EXPECT_EQ(0, scale[2000]);
//    EXPECT_EQ(0, precision[3999]);
//    EXPECT_EQ(0, scale[3999]);

    EXPECT_EQ(1, precision[4000]);
    EXPECT_EQ(1, scale[4000]);
    EXPECT_EQ(10001001, precision[5000]);
    EXPECT_EQ(4, scale[5000]);
    EXPECT_EQ(19992, precision[5999]);
    EXPECT_EQ(1, scale[5999]);

    free(bool_exact);
}

#if 1
TEST(orc_reader_inspection, testPredicatePushdown_orc) {
    const char* filename = "examples/TestOrcFile.testPredicatePushdown.orc";

    CudaOrcReaderImplProto* reader = (CudaOrcReaderImplProto*)gdf_create_orc_reader();

    ASSERT_EQ(GDF_ORC_SUCCESS, reader->ReadFromFile(filename));
    const auto ft = reader->getFooter();
    const int numRows = 3500;

    EXPECT_EQ(numRows, ft->numberofrows);
    EXPECT_EQ(1, ft->stripes_size);
    EXPECT_EQ(3, ft->types_size);   // struct, int, string

    orc_sint32* gdf_int = reinterpret_cast<orc_sint32*>(reader->getHost(1).stream_data);
#if 0
    EXPECT_EQ(0, gdf_int[0]);
    EXPECT_EQ(300, gdf_int[1]);
    EXPECT_EQ(600, gdf_int[2]);
#else
    for (int i = 0; i < numRows; i++) {
        EXPECT_EQ(300 * i, gdf_int[i]);
    }
#endif

    // checking string direct mode 
    orc_byte* gdf_char = reinterpret_cast<orc_byte*>(reader->getHost(2).stream_data);
    orc_sint32* gdf_length = reinterpret_cast<orc_sint32*>(reader->getHost(2).stream_length);
    EXPECT_EQ('0', gdf_char[0]);
    EXPECT_EQ('a', gdf_char[1]);
    EXPECT_EQ('1', gdf_char[2]);
    EXPECT_EQ('4', gdf_char[3]);

    EXPECT_EQ(1, gdf_length[0]);
    EXPECT_EQ(1, gdf_length[1]);
    EXPECT_EQ(2, gdf_length[2]);
    EXPECT_EQ(2, gdf_length[3]);


}
#endif

#if 1
TEST(orc_reader_inspection, nested_null) {
    const char* filename = "examples/TestOrcFile.testUnionAndTimestamp.orc";

    CudaOrcReaderImplProto* reader = (CudaOrcReaderImplProto*)gdf_create_orc_reader();

    ASSERT_EQ(GDF_ORC_SUCCESS, reader->ReadFromFile(filename));
    const auto ft = reader->getFooter();

    // -- validate the postscript/footer info
    EXPECT_EQ(270, ft->footerLength);
    EXPECT_EQ(OrcCompressionKind::NONE, ft->compressionKind);
    EXPECT_EQ(0, ft->compressionBlockSize);
    EXPECT_EQ(234, ft->metadataLength);

    EXPECT_EQ(3, ft->headerlength);
    EXPECT_EQ(20906, ft->contentlength);
    EXPECT_EQ(5077, ft->numberofrows);
    EXPECT_EQ(10000, ft->rowindexstride);
    EXPECT_EQ(0, ft->user_metadata_size);
    EXPECT_EQ(6, ft->statistics_size);
    EXPECT_EQ(6, ft->types_size);
    EXPECT_EQ(2, ft->stripes_size);


    orc_bitmap* presents[6];
    for (int i = 0; i < 6; i++) {
        presents[i] = reinterpret_cast<orc_bitmap*>(reader->getHost(i).stream_present);
    }

    EXPECT_EQ(NULL, presents[0]);
    EXPECT_EQ(0xe3, presents[1][0]);
    EXPECT_EQ(0xfb, presents[2][0]);
    EXPECT_EQ(0xf9, presents[3][0]);    // actually this value is wrong, union is not supported
    EXPECT_EQ(0xf9, presents[4][0]);    // actually this value is wrong, union is not supported
    EXPECT_EQ(0xe3, presents[5][0]);

    EXPECT_EQ(0x03, presents[1][9]);
    EXPECT_EQ(0x03, presents[5][9]);

    EXPECT_EQ(0x00, presents[1][634]);
    EXPECT_EQ(0xff, presents[2][634]);
//    EXPECT_EQ(0x00, presents[3][634]);    // this should be zero.
    EXPECT_EQ(0xff, presents[4][634]);
    EXPECT_EQ(0x00, presents[5][634]);

#if 0
    orc_uint32* gdf_ns = reinterpret_cast<orc_uint32*>(reader->getHost(1).stream_secondary);
    EXPECT_EQ(197<<3 +5, gdf_ns[6]);


    orc_sint64* gdf_ts = reinterpret_cast<orc_sint64*>(reader->getHost(1).stream_gdf_timestamp);

    EXPECT_EQ(convertGdfTimestampMs(2000, 3, 12, 15, 00, 00, 0), gdf_ts[0]);
    EXPECT_EQ(convertGdfTimestampMs(2000, 3, 20, 12, 00, 00, 123), gdf_ts[1]);
    EXPECT_EQ(convertGdfTimestampMs(1970, 1, 1, 0, 00, 00, 0), gdf_ts[5]);
    EXPECT_EQ(convertGdfTimestampMs(1970, 5, 5, 12, 34, 56, 197), gdf_ts[6]);
    EXPECT_EQ(convertGdfTimestampMs(1971, 5, 5, 12, 34, 56, 197), gdf_ts[7]);
    EXPECT_EQ(convertGdfTimestampMs(1973, 5, 5, 12, 34, 56, 197), gdf_ts[8]);
    EXPECT_EQ(convertGdfTimestampMs(2037, 5, 5, 12, 34, 56, 203), gdf_ts[73]);
#endif

}
#endif

#if 0
TEST(orc_reader_inspection, zipLoad) {
    const char* filename = "examples/demo-12-zlib.orc";

//    CudaOrcReader* reader = gdf_create_orc_reader();

//    ASSERT_EQ(GDF_ORC_SUCCESS, reader->ReadFromFile(filename));
//    const auto ft = reader.getFooter();

}
#endif

#if 1    // not well tested.
TEST(orc_reader_inspection, string_dict) {
    const char* filename = "examples/orc_split_elim.orc";

    CudaOrcReaderImplProto* reader = (CudaOrcReaderImplProto*)gdf_create_orc_reader();

    ASSERT_EQ(GDF_ORC_SUCCESS, reader->ReadFromFile(filename));
    const auto ft = reader->getFooter();
    
    int string_id = 2;

    orc_byte* archive = reinterpret_cast<orc_byte*>(reader->getHost(string_id).stream_data);
    orc_uint32* length = reinterpret_cast<orc_uint32*>(reader->getHost(string_id).stream_length);
    orc_uint32* index = reinterpret_cast<orc_uint32*>(reader->getHost(string_id).stream_dictionary_data);

    orc_byte ref[] = "foozebra";
    orc_uint32 len[] = {3,5};
}
#endif

#endif  // #ifndef GDF_ORC_NO_FILE_TEST
