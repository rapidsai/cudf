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

#include <stdio.h>
#include <string.h>

#include "kernel_orc.cuh"
#include "orc_proto.hpp"
#include "orc_read_impl_proto.h"
#include "orc_memory.h"

#if _DEBUG
#define ENABLE_DUMP
#endif

CudaOrcReaderImplProto::~CudaOrcReaderImplProto()
{ 
    release();
};

void CudaOrcReaderImplProto::release() 
{
    orc_file.release();
    file_top = NULL;

    DeAllocateArrays();    // deallocate ORC arrays, detached buffers won't be released.
}

void CudaOrcReaderImplProto::GetPSInfo(const orc::proto::PostScript& ps)
{
    footer_info.footerLength = ps.footerlength();
    footer_info.compressionKind = OrcCompressionKind(ps.compression());
    footer_info.compressionBlockSize = ps.compressionblocksize();
    footer_info.metadataLength = ps.metadatalength();
    if (ps.version_size() >= 2) {
        footer_info.versions[0] = ps.version(0);
        footer_info.versions[1] = ps.version(1);
    }
    else {
        footer_info.versions[0] = footer_info.versions[1] = 0;
    }

    footer_info.writerVersion = ps.writerversion();
    is_no_compression = (footer_info.compressionKind == OrcCompressionKind::NONE) ? true : false;
    // PRINTF("magic   : %s", ps.magic().c_str());    // check postscript tail. ORC postsript must end of "ORC"
}

void CudaOrcReaderImplProto::GetFooterInfo(const orc::proto::Footer& ft) {
    footer_info.headerlength = ft.headerlength();
    footer_info.contentlength = ft.contentlength();
    footer_info.numberofrows = ft.numberofrows();
    footer_info.rowindexstride = ft.rowindexstride();
    footer_info.user_metadata_size = ft.metadata_size();
    footer_info.statistics_size = ft.statistics_size();
    footer_info.types_size = ft.types_size();
    footer_info.stripes_size = ft.stripes_size();
    
    D_MSG(" number of record : %d.", footer_info.numberofrows);
}

void CudaOrcReaderImplProto::GetStripesInfo(const orc::proto::Footer& ft)
{
    size_t startRowIndex = 0;

    // get each stripe information!
    int stripes_size = ft.stripes_size();
    stripes.resize(stripes_size);
    for (int i = 0; i < stripes_size; i++) {
        const auto& info = ft.stripes(i);

        stripes[i].offset = info.offset();
        stripes[i].indexLength = info.indexlength();
        stripes[i].dataLength = info.datalength();
        stripes[i].footerLength = info.footerlength();
        stripes[i].numberOfRows = info.numberofrows();
        stripes[i].startRowIndex = startRowIndex;

        startRowIndex += info.numberofrows();
    }
}


void CudaOrcReaderImplProto::GetTypesInfo(const orc::proto::Footer& ft)
{
    assert(ft.statistics_size()  == ft.types_size());
    D_MSG("type count  :%d", ft.types_size());
    D_MSG("stat count  :%d", ft.statistics_size());

    int types_size = ft.types_size();
    types.resize(types_size);
    for (int i = 0; i < types_size; i++) {
        const auto& type = ft.types(i);
        OrcTypeInfo& ti = types[i];

        ti.kind = ORCTypeKind(type.kind());
        ti.subtypes_size = type.subtypes_size();
        int field_name_count = type.fieldnames_size();
        for (int k = 0; k < ti.subtypes_size; k++) {
            int id = type.subtypes(k);
            OrcTypeInfo& subtype = types[id];
            subtype.parent_id = i;
            if( k < field_name_count)subtype.fieldNames = type.fieldnames(k);
        }

        if (type.has_maximumlength()) ti.maximumLength = type.maximumlength();
        
        if (type.has_precision()) ti.precision = type.precision();
        if (type.has_scale()) ti.scale = type.scale();


        switch (ti.kind) {
        case OrcShort:      ti.elementType = OrcElementType::Sint16; break;
        case OrcInt:        ti.elementType = OrcElementType::Sint32; break;
        case OrcLong:       ti.elementType = OrcElementType::Sint64; break;
        case OrcFloat:      ti.elementType = OrcElementType::Float32; break;
        case OrcDouble:     ti.elementType = OrcElementType::Float64; break;
        case OrcTimestamp:  ti.elementType = OrcElementType::Sint64; break;
        case OrcDate:       ti.elementType = OrcElementType::Sint64; break;
        case OrcDecimal:    ti.elementType = OrcElementType::Sint64; break;
            break;
        }

        {
            const auto& stat = ft.statistics(i);
            ti.nonNullDataCount = stat.numberofvalues();
            ti.nullDataCount = footer_info.numberofrows - ti.nonNullDataCount;
            
            if (stat.has_hasnull()) {
                ti.hasNull = (stat.hasnull()) ? 1 : 0;
            }
            else {
                ti.hasNull = -1;
            }

            if (stat.has_stringstatistics()) ti.variableLengthCount = stat.stringstatistics().sum();
            if (stat.has_binarystatistics()) ti.variableLengthCount = stat.binarystatistics().sum();
        }
    }

#if 1
    if (ft.statistics_size()) {
        D_MSG("Footer ColumnStatistics: %d", ft.statistics_size());
        D_MSG("  [id]: num Values, hasNULL, bytesOnDisk");

    }
    for (int i = 0; i < ft.statistics_size(); i++) {
        const auto& stat = ft.statistics(i);
        D_MSG("  [%d]: %d, %d, %d", i, stat.numberofvalues(), types[i].hasNull, stat.bytesondisk());
    }
#endif
}

bool CudaOrcReaderImplProto::isVariableContainer(int id)
{
    if (id < 0)return false;
    assert(id < types.size());

    switch (types[id].kind) {
    case OrcList:
    case OrcMap:
        return true;
    }

    return isVariableContainer(types[id].parent_id);
}

bool CudaOrcReaderImplProto::isVariableLength(int id)
{
    assert(id < types.size());

    switch (types[id].kind) {
    case OrcString:
    case OrcBinary:
    case OrcVarchar:
    case OrcChar:
        return true;
    }

    return isVariableContainer(types[id].parent_id);
}

bool CudaOrcReaderImplProto::ValidateColumns()
{
    bool retval = true;
    for (int i = 0; i < types.size(); i++) {
        OrcTypeInfo& ti = types[i];

        ti.isVariableLength = isVariableLength(i);

#if 1   // skip the data load for now
        if (ti.isVariableLength)ti.skipDataLoad = true;
#endif
        if (ti.isVariableLength)
        {   
            if (ti.isValidated)continue;

            // if the column is variable length, check it has total length and the length info for each stripe
            if (ti.variableLengthCount < 0) {
                retval = false; 
                continue;
            }
            if (ti.varLengthInfo.size() != footer_info.stripes_size) { 
                retval = false;
                continue;
            }

            int the_length_count = 0;
            auto& info = ti.varLengthInfo;
            for (int k = 0; k < footer_info.stripes_size; k++) {
                if (info[k].numberOfRows < 0) { // numberOfRows = -1 if stripe does not have the info
                    retval = false; 
                    break;
                }
                info[k].startRowIndex = the_length_count;
                the_length_count += info[k].numberOfRows;
                D_MSG("[%d:%d]: (%d, %d)\n", i, k, info[k].numberOfRows, info[k].startRowIndex);
            }
            if (retval == true && the_length_count == ti.variableLengthCount)
            {
                ti.isValidated = true;
                ti.skipDataLoad = false;
            }
            else {
                retval = false;
            }

        }
        else {
            switch (ti.kind) {
            case OrcDecimal:
                break;
            case OrcTimestamp:
            case OrcDate:                
                break;
            default:
                break;
            }
        }

        if (ti.hasNull == 0) {
            ti.skipPresentation = true;
        }
    }


    return retval;
}

void CudaOrcReaderImplProto::GetMetaDataInfo(const orc::proto::Metadata& mt)
{
    D_MSG("Metadata ColumnStatistics:");
    D_MSG("  [stripe, column]: num Values, hasNULL, bytesOnDisk");

    // allocate varLengthInfo
    for (int i = 0; i < types.size(); i++) {
        OrcTypeInfo& ti = types[i];
        if (ti.isVariableLength) {
            ti.varLengthInfo.resize(footer_info.stripes_size);
        }
    }

    for (int i = 0; i < mt.stripestats_size(); i++) {
        auto stripe_stats = mt.stripestats(i);
        for (int k = 0; k < stripe_stats.colstats_size(); k++) {
            const auto& stat = stripe_stats.colstats(k);

            OrcTypeInfo& ti = types[k];
            if (ti.isVariableLength) {  // get var length
                ti.varLengthInfo[i].numberOfRows = -1;

                switch (ti.kind) {
                case ORCTypeKind::OrcString:
                case ORCTypeKind::OrcVarchar:
                case ORCTypeKind::OrcChar:
                    if (stat.has_stringstatistics() && stat.stringstatistics().has_sum()) {
                        ti.varLengthInfo[i].numberOfRows = stat.stringstatistics().sum();
                    }
                    break;
                case ORCTypeKind::OrcBinary:
                    if (stat.has_binarystatistics() && stat.binarystatistics().has_sum()) {
                        ti.varLengthInfo[i].numberOfRows = stat.binarystatistics().sum();
                    }
                    break;
                default:
                    break;
                }
            }

            D_MSG("  [%d,%d]: %d, %d, %d", i, k, stat.numberofvalues(), stat.hasnull(), stat.bytesondisk());
        }
    }

}


void CudaOrcReaderImplProto::SelectColumns()
{

}

CudaOrcError_t CudaOrcReaderImplProto::ReadFromFile(const char* filename)
{
    size_t file_size = 0;
    ORC_RETURN_IF_ERROR(orc_file.loadFile(file_top, file_size, filename));

    D_MSG("file size: %d\n", file_size);
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // initialize OrcMem class
    OrcMem::OrcMemArgument memArg;
    memArg.blockSize = 2 * 1024 * 1024; // 2 MB for now
    OrcMem::InitializeOrcMem(memArg);

    // check file header. ORC must begin from "ORC".
    if (strncmp("ORC", (const char*)file_top, 3)) {
        D_MSG("Invalid input file: <%s>\n", filename);
        D_MSG("The file must be started with 3 chars: [ORC] ");
        return GDF_ORC_INVALID_FILE_FORMAT;
    }

    // get Postscript data size at the last byte of the file.
    int Postscript_size = (unsigned char)(file_top[file_size - 1]);
    D_MSG("Postscript_size: %d\n", Postscript_size);

    // parse postscipt
    const unsigned char* postsript = file_top + file_size - 1 - Postscript_size;
    orc::proto::PostScript ps;
    if (!ps.ParseFromArray((void*)postsript, Postscript_size))
    {
        D_MSG("*** fail to parse postscript.");
        ORC_SET_ERROR_AND_RETURN(GDF_ORC_INVALID_FILE_FORMAT_PROTOBUF_FAILURE);
    }

    GetPSInfo(ps);

    // return UNSUPPORTED_COMPRESSION_TYPE if comp kind is not of none or zlib
    if (footer_info.compressionKind != OrcCompressionKind::NONE &&
        footer_info.compressionKind != OrcCompressionKind::ZLIB) {
        PRINTF("The compression mode is not supported: [%d]", footer_info.compressionKind);
        ORC_SET_ERROR_AND_RETURN(GDF_ORC_UNSUPPORTED_COMPRESSION_TYPE);
    }

    OrcCompressionKind compKind = footer_info.compressionKind;
    CompressedProtoLoader<orc::proto::Footer> ft(compKind);

    ORC_RETURN_IF_ERROR(ft.ParseFromArray(postsript - ps.footerlength(), footer_info.footerLength));

    GetFooterInfo(ft.get());
    GetStripesInfo(ft.get());
    GetTypesInfo(ft.get());

    // validate the columns. Returns false if any of columns has variable length member
    bool isValid = ValidateColumns();

    if (footer_info.metadataLength == 0 ) {
        hasMetadata = false;
        D_MSG("No metadata.");
    }
    else if (!isValid)  // try to read metadata if any of columns is kinda variable length
    {    
        CompressedProtoLoader<orc::proto::Metadata> mt(compKind);
        const orc_byte *pos = postsript - footer_info.footerLength - footer_info.metadataLength;
        ORC_RETURN_IF_ERROR(mt.ParseFromArray(pos, footer_info.metadataLength));
        hasMetadata = true;

        GetMetaDataInfo(mt.get());

        isValid = ValidateColumns();

        dumpMetaData(mt.get());
    }

#if 1
    if (!isValid) {
        D_MSG("*****************************************\n");
        D_MSG("    the columns are not validated yet!!!!\n");
        D_MSG("*****************************************\n");
//        ORC_SET_ERROR_AND_RETURN(GDF_ORC_UNSUPPORTED_DATA_TYPE);
    }
#endif

    dumpFooterInfo();
    dumpStripes();
    dumpTypes();

    // if allocation is failed, returns with error code and no decode at GPU.
    ORC_RETURN_IF_ERROR(AllocateArrays());

    // this is main of decoding ORC data at GPU
    Decode();

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    orc_file.release();

    return status;
}

orc_uint32 CudaOrcReaderImplProto::findGMToffset(const char* region)
{
    // early return if there is convertToGMT = false or no region or GMT.
    if ( ! option.convertToGMT ) return 0;
    if (region == NULL || region[0] == 0) return 0;
    if (!strcmp("GMT", region))return 0;

    int offset;
    auto it = timezoneCache.find(region);    // find from cache
    if (it != timezoneCache.end())
    {
        offset = it->second;
    }
    else {
        bool ret = findGMToffsetFromRegion(offset, region);
        if (!ret) {
            PRINTF("The timezone <%s> is not found.\n", region);
        }

        // insert to cache even if findGMToffsetFromRegion() failed.
        timezoneCache.insert(std::make_pair(std::string(region), offset));
    }

    return offset;
}

void CudaOrcReaderImplProto::ParseStripeColumns(const orc::proto::StripeFooter& stripe, OrcStripeArguemnts& stripeArg)
{
    assert(types.size() == stripe.columns_size());
    stripeArg.setNumOfColumn(stripe.columns_size());

    for (int i = 0; i < types.size(); i++) {
        const auto& col = stripe.columns(i);
        auto& colArg = stripeArg.getColumnArg(i);

        colArg.encoding = ORCColumnEncodingKind(col.kind());
        colArg.dictionary_size = col.dictionarysize();
        colArg.bloom_encoding = col.bloomencoding();
        colArg.type = & types[i];
    }

    if (stripe.has_writertimezone())
    {
        stripeArg.SetGMToffset(findGMToffset(stripe.writertimezone().c_str()));
    }
    else {
        stripeArg.SetGMToffset(0);
    }

}

int ElementSize(OrcElementType type) {
    switch (type) {
    case OrcElementType::Uint16:
    case OrcElementType::Sint16:  return 2;
    case OrcElementType::Uint32:
    case OrcElementType::Sint32:  return 4;
    case OrcElementType::Uint64:
    case OrcElementType::Sint64:  return 8;
    case OrcElementType::Float32: return 4;
    case OrcElementType::Float64: return 8;
    default:
        return 1;
    }
}

void CudaOrcReaderImplProto::ParseStripeStreams(const orc::proto::StripeFooter& ft, OrcStripeArguemnts& stripeArg)
{
    // the number of stream varies at each stripe
    stripeArg.setNumOfStream(ft.streams_size());

    size_t offset_from_stripe = 0;
    for (int k = 0; k < ft.streams_size(); k++) {
        const auto stream = ft.streams(k);
        // set source stream range
        stripeArg.setStreamRange(ORCStreamKind(stream.kind()), k,
            stream.column(), offset_from_stripe, stream.length());
        offset_from_stripe += stream.length();

        auto& streamArg = stripeArg.getStreamArg(k);
        auto& columnArg = stripeArg.getColumnArg(stream.column());
        ORCstream& strm = deviceArray[stream.column()];

        auto& info = stripes[stripeArg.StripeId()];

        int the_start_index    = info.startRowIndex;
        int the_number_of_rows = info.numberOfRows;

        if (columnArg.type->isVariableLength) {
            auto& varLenInfo = columnArg.type->varLengthInfo[stripeArg.StripeId()];
            the_start_index    = varLenInfo.startRowIndex;
            the_number_of_rows = varLenInfo.numberOfRows;
        }
        bool is_dictionary_encoded = false;
        if (columnArg.encoding == ORCColumnEncodingKind::OrcDictionary ||
            columnArg.encoding == ORCColumnEncodingKind::OrcDictionary_V2)
        {
            is_dictionary_encoded = true;
        }

        // set output range. 
        switch (streamArg.kind()) {
        case OrcRowIndex:
        case OrcBloomFilter:
        case OrcBloomFilterUtf8:
            // to nothing for now.
            break;
        case OrcPresent:
            streamArg.SetPresentTarget(strm.stream_present, info.startRowIndex, info.numberOfRows);
            columnArg.stream_present = &streamArg;
            break;
        case OrcData:
            if (is_dictionary_encoded) {
                // if the data stream is encoded by dictionary
                // the gdf orc reader swaps the data and dictionary_data streams for the consistency of column stream type
                streamArg.SetTarget(strm.stream_dictionary_data, info.startRowIndex, info.numberOfRows,
                    ElementSize(columnArg.type->elementType));
                columnArg.stream_dictionary_data = &streamArg;
            }
            else {
                streamArg.SetTarget(strm.stream_data, the_start_index, the_number_of_rows,
                    ElementSize(columnArg.type->elementType));
                columnArg.stream_data = &streamArg;
            }

            break;
        case OrcLength:
            if (is_dictionary_encoded) {
                // the output length is columnArg.dictionary_size (count of the dictionaries)
                streamArg.SetTarget(strm.stream_length, info.startRowIndex, columnArg.dictionary_size, 4);
            }
            else {
                streamArg.SetTarget(strm.stream_length, info.startRowIndex, info.numberOfRows, 4);
            }
            columnArg.stream_length = &streamArg;
            break;
        case OrcSecondary:
        {
            int element_size = 4;
            if (columnArg.type->kind == ORCTypeKind::OrcTimestamp) element_size = GDF_ORC_TIMESTAMP_NANO_PRECISION;
            streamArg.SetTarget(strm.stream_secondary, the_start_index, the_number_of_rows, element_size);
            columnArg.stream_second = &streamArg;
            break;
        }
        case OrcDictionaryCount:
            EXIT("this is no longer supported");
            break;
        case OrcDictionaryData:
            // since the data stream is decoded into stream_dictionary_data,
            // OrcDictionaryData stream is decoded int stream_data.
            streamArg.SetTarget(strm.stream_data, the_start_index, the_number_of_rows,
                ElementSize(columnArg.type->elementType));
            columnArg.stream_data = &streamArg;

            break;
        default:
            // Todo
            break;
        }
    }    // end of for loop
}

void CudaOrcReaderImplProto::Decode()
{
    if (footer_info.stripes_size < 1) return;

    if (footer_info.compressionKind != OrcCompressionKind::NONE && 
        footer_info.compressionKind != OrcCompressionKind::ZLIB
        )return;

    bool isCompressed = false;
    int released_count;
    released_count = 0;
    stripeArgs.resize(stripes.size());

    for (int i = 0; i < stripes.size(); i++) {
        OrcStripeInfo& info = stripes[i];
        auto& stripeArg = stripeArgs[i];

        stripeArg.SetCompressionKind(footer_info.compressionKind);
        stripeArg.SetStripeInfo(&info, i);
        stripeArg.SetDeviceArray(&deviceArray);

        CompressedProtoLoader<orc::proto::StripeFooter> spf(footer_info.compressionKind);
        const orc_byte* stripe_footer = file_top + info.offset + info.indexLength + info.dataLength;
        status = spf.ParseFromArray(stripe_footer, info.footerLength);

        // allocate and copy stripe info into GPU.
        stripeArg.allocateAndCopyStripe(file_top + info.offset, info.indexLength + info.dataLength);

        ParseStripeColumns(spf.get(), stripeArg);    // parse StripeFooter.columns
        ParseStripeStreams(spf.get(), stripeArg);    // parse StripeFooter.streams

        stripeArg.DecodePresentStreams();           // decode present streams.
        stripeArg.DecodeDataStreams();              // decode data streams.

#ifdef _DEBUG
        if (i == 0) dumpStreamInfo(spf.get());
#endif

#if 0
        if (stripeArgs[released_count].isFinished())
        {
            stripeArgs[released_count].releaseStripeBuffer();
            released_count++;
        }
#endif
    }

    // force sync for now
    CudaFuncCall(cudaGetLastError());
    CudaFuncCall(cudaDeviceSynchronize());
    
    // free all stripes
    for (int i = released_count; i < stripes.size(); i++) {
        if (stripeArgs[i].isFinished())
        {
            stripeArgs[i].releaseStripeBuffer();
        }
    }
}

CudaOrcError_t orcCudaAlloc(void** devPtr, size_t size) {
    if (size) {
        CudaFuncCall(cudaMallocManaged(devPtr, size, cudaMemAttachGlobal));
    }
    else {
        *devPtr = NULL;
    }
    return GDF_ORC_SUCCESS;
}

CudaOrcError_t CudaOrcReaderImplProto::AllocateArrays()
{
    deviceArray.resize(footer_info.types_size);

    for (int i = 0; i < footer_info.types_size; i++) {
        const auto &ti = types[i];
        ORCstream& strm = deviceArray[i];
        memset(&strm, 0, sizeof(ORCstream));    // fill by NULL

        bool has_data = true;
        int  element_size = 0;
        bool has_present = true;

        switch (ti.kind)
        {
        case OrcBolean:     element_size = 1; break;
        case OrcByte:       element_size = 1; break;
        case OrcShort:      element_size = 2; break;
        case OrcInt:        element_size = 4; break;
        case OrcLong:       element_size = 8; break;
        case OrcFloat:      element_size = 4; break;
        case OrcDouble:     element_size = 8; break;
        case OrcVarchar:
        case OrcChar:
        case OrcString:
            if (ti.variableLengthCount > 0) {
                orcCudaAlloc((void**)& strm.stream_data, ti.variableLengthCount);
                orcCudaAlloc((void**)& strm.stream_length, footer_info.numberofrows * 4);
                orcCudaAlloc((void**)& strm.stream_dictionary_data, footer_info.numberofrows * 4);
                orcCudaAlloc((void**)& strm.stream_gdf_string, footer_info.numberofrows * sizeof(gdf_string));
            }
            has_data = false;
            break;
        case OrcBinary:
            // binary doesn't have dictionary stream.
            if (ti.variableLengthCount > 0) {
                orcCudaAlloc((void**)& strm.stream_data, ti.variableLengthCount);
                orcCudaAlloc((void**)& strm.stream_length, footer_info.numberofrows * 4);
                orcCudaAlloc((void**)& strm.stream_gdf_string, footer_info.numberofrows * sizeof(gdf_string));
            }
            has_data = false;
            break;
        case OrcTimestamp:
            // tbd.
            element_size = 8;
            orcCudaAlloc((void**)& strm.stream_secondary, footer_info.numberofrows * GDF_ORC_TIMESTAMP_NANO_PRECISION);
            orcCudaAlloc((void**)& strm.stream_gdf_timestamp, footer_info.numberofrows * 8);   // this stream is exposed into gdf
            break;
        case OrcList:
        case OrcMap:
        case OrcStruct:
        case OrcUnion:
            has_data = false;
            // those are manages the data structure
            break;
        case OrcDecimal:
            // these info can be used.
            //            if (type.has_precision()) ti.precision = type.precision();
            //            if (type.has_scale()) ti.scale = type.scale();
            element_size = 8;
            orcCudaAlloc((void**)& strm.stream_secondary, footer_info.numberofrows * 4);
            break;
        case OrcDate:
            element_size = 8;
            break;
        default:
            D_MSG("Unknown Data type\n");
            has_data = false;
            break;
        }    // switch (ti.kind)

        // all types may have stream_present.
        if (has_data && element_size) {
            orcCudaAlloc((void**)& (strm.stream_data), footer_info.numberofrows * element_size);
        }

        if (!ti.skipPresentation) {
            orcCudaAlloc((void**)& (strm.stream_present), (footer_info.numberofrows + 7) >> 3);
        }
    }
    CudaFuncCall(cudaDeviceSynchronize());
    return GDF_ORC_SUCCESS;
}

CudaOrcError_t CudaOrcReaderImplProto::DeAllocateArrays()
{
#define ORC_SAFE_FREE( mem  ) if( mem  ){  CudaFuncCall(cudaFree(mem)); mem = NULL; }

    for (int i = 0; i < deviceArray.size(); i++) {
        ORCstream& stream = deviceArray[i];

        ORC_SAFE_FREE(stream.stream_present);
        ORC_SAFE_FREE(stream.stream_data);
        ORC_SAFE_FREE(stream.stream_length);        // unioned with stream_secondary
        ORC_SAFE_FREE(stream.stream_dictionary_data);
        ORC_SAFE_FREE(stream.stream_secondary);
        ORC_SAFE_FREE(stream.stream_gdf_string);    // unioned with stream_gdf_timestamp
    
    }
    CudaFuncCall(cudaDeviceSynchronize());
#undef ORC_SAFE_FREE
    return GDF_ORC_SUCCESS;
}

const ORCstream& CudaOrcReaderImplProto::getHost(int i) {
    return deviceArray[i];
}

void CudaOrcReaderImplProto::dumpFooterInfo()
{
#ifdef ENABLE_DUMP
#define DUMP(name) PRINTF("  " #name " : %d", footer_info. name)

    PRINTF("Footer info:");

    DUMP(headerlength);
    DUMP(contentlength);
    DUMP(numberofrows);
    DUMP(rowindexstride);
    DUMP(user_metadata_size);
    DUMP(statistics_size);
    DUMP(types_size);
    DUMP(stripes_size);
    DUMP(compressionBlockSize);

    const char* writer_name = "undefined";
    const char* writer[] = { "original", "HIVE-8732", "HIVE-4243", "HIVE-12055", "HIVE-13083", "ORC-101", "ORC-105" };
    if (footer_info.writerVersion >= 0 && footer_info.writerVersion <= 6) {
        writer_name = writer[footer_info.writerVersion];
    }

    PRINTF("  version = %s [%d.%d]", writer_name, footer_info.versions[0], footer_info.versions[1]);
    
    const char* compressionKindName[] = { "NONE", "ZLIB", "SNAPPY", "LZO", "LZ4", "ZSTD" };
    PRINTF("  comp kind : %s", compressionKindName[footer_info.compressionKind]);
#undef DUMP
#endif
}

void CudaOrcReaderImplProto::dumpStripes()
{
#ifdef ENABLE_DUMP
    PRINTF("Stripe info: %d", footer_info.stripes_size);
    PRINTF("  [num]: offset, index, data, footer length, num of rows");
    for (int i = 0; i < footer_info.stripes_size; i++) {
        const auto &s = stripes[i];
        PRINTF("  [%d]: %d, %d, %d, %d, %d", i, s.offset, s.indexLength, s.dataLength, s.footerLength, s.numberOfRows);
    }
#endif
}

#ifdef ENABLE_DUMP
const char* typeKindName[] = { "BOOLEAN", "BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "STRING", "BINARY",
"TIMESTAMP", "LIST", "MAP", "STRUCT", "UNION", "DECIMAL", "DATE", "VARCHAR", "CHAR"
};
#endif

void CudaOrcReaderImplProto::dumpTypes()
{
#ifdef ENABLE_DUMP

    PRINTF("Types info: %d", footer_info.types_size);
    PRINTF("  [num, parent id]: kind id, data count (hasNull, length), name");

    for (int i = 0; i < footer_info.types_size; i++) {
        const auto &t = types[i];

        //        PRINTF("  [%d]: %d, %d, %s", i, t.kind, t.subtypes_size, t.fieldNames.c_str());
        PRINTF("  [%d, %d]: %s, %d (%d, %d), %s", i, t.parent_id, typeKindName[t.kind],
            t.nonNullDataCount, t.hasNull, t.variableLengthCount, t.fieldNames.c_str());
    }
#endif
}

void CudaOrcReaderImplProto::dumpMetaData(const orc::proto::Metadata& mt)
{
#ifdef ENABLE_DUMP
    D_MSG("Metadata ColumnStatistics:");
    D_MSG("  [stripe, column]: num Values, hasNULL, bytesOnDisk");

    for (int i = 0; i < mt.stripestats_size(); i++) {
        auto stripe_stats = mt.stripestats(i);
        for (int k = 0; k < stripe_stats.colstats_size(); k++) {
            const auto& stat = stripe_stats.colstats(k);
            D_MSG("  [%d,%d]: %d, %d, %d", i, k, stat.numberofvalues(), stat.hasnull(), stat.bytesondisk());

        }
    }
#endif
}

void CudaOrcReaderImplProto::dumpStreamInfo(const orc::proto::StripeFooter& ft)
{
#ifdef ENABLE_DUMP
    const char* streamKindName[] = { "PRESENT", "DATA", "LENGTH", "DICTIONARY_DATA", "DICTIONARY_COUNT", "SECONDARY", "ROW_INDEX", "BLOOM_FILTER", "BLOOM_FILTER_UTF8" };
    const char* columnEncodingsKindName[] = { "DIRECT", "DICTIONARY", "DIRECT_V2", "DICTIONARY_V2" };

    PRINTF("stream count: %d", ft.streams_size());
    for (int k = 0; k < ft.streams_size(); k++) {
        const auto& stream = ft.streams(k);
        int id = stream.column();
        const auto& type = types[id];

        PRINTF("[%d,%d]: %s, %s, %s", 
            k, id, streamKindName[stream.kind()], columnEncodingsKindName[ft.columns(id).kind()], typeKindName[type.kind]);
    }

    if (ft.has_writertimezone()) {
        PRINTF("writer time zone: %s", ft.writertimezone().c_str());
    }


    PRINTF(" ");
#endif
}

// ----------------------------------------------------------------------------------

