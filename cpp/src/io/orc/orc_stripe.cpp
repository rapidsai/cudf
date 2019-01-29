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

#include "orc_stripe.h"
#include "kernel_orc.cuh"    // need for cuda
#include "orc_memory.h"
#include <assert.h>

bool OrcStripeArguemnts::isFinished() 
{
    // for now, force synchronizing the devices
    CudaFuncCall(cudaGetLastError());
    CudaFuncCall(cudaDeviceSynchronize());

    return true; 
};    

void OrcStripeArguemnts::releaseStripeBuffer()
{
    CudaFuncCall(cudaFree((void*)stride_addr_gpu));
    stride_addr_gpu = NULL;
}


CudaOrcError_t OrcStripeArguemnts::allocateAndCopyStripe(const orc_byte* src, size_t size) {
    stride_addr_cpu = src;
    stride_length = size;

    // the cuda memory size must be multipled by 4 for orc reader for cuda
    size_t local_size = (size + 0x3) ^ 0x3;

    cudaError_t error = cudaMalloc((void**)&stride_addr_gpu, local_size);
    if (error != cudaSuccess)
    {
        return (error == cudaErrorMemoryAllocation) ? GDF_ORC_OUT_OF_MEMORY : GDF_ORC_INVALID_API_CALL;
    }
    error = cudaMemcpy((void*)stride_addr_gpu, (void*)stride_addr_cpu, size, cudaMemcpyHostToDevice);
    return (error == cudaSuccess) ? GDF_ORC_SUCCESS : GDF_ORC_INVALID_API_CALL;
}

int OrcStripeArguemnts::FindValidParant(int id)
{
    if (id < 0) return -1;

    auto& col = columnArgs[id];
    int parent_id = col.type->parent_id;
    if (parent_id < 0) return -1;

    auto& parent = columnArgs[parent_id];
    if (parent.stream_present)return parent_id;

    if (parent.type->kind == OrcUnion) {
        // if the parent is union, present stream is also affected by tag stream of the union.
        return parent_id;
    }

    return FindValidParant(parent_id);
}

void OrcStripeArguemnts::DecodeCompressedStream(KernelParamBase *param, OrcStreamArguemnts *stream)
{
    if (compKind == OrcCompressionKind::NONE)return;

    // the stream is reprenented as chunks when ORC is compressed.
    OrcChunkDecoder chunkDecoder;
    chunkDecoder.SetCompressionKind(compKind);

    const orc_byte* source = stream->sourceCpuAddr();
    size_t size = stream->sourceSize();

    // treat the stream as uncompressed if the stream is single chunk and uncompressed, 
    if (chunkDecoder.IsSingleUncompressed(source, size))
    {
        param->input += 3;
        param->input_size -= 3;
        return;
    }

    std::vector<OrcBuffer> decodedChunks;
    CudaOrcError_t ret = chunkDecoder.Decode(decodedChunks, &holder, source, param->input, size);
    int totalSize = 0;
    for (OrcBuffer buf : decodedChunks)totalSize += buf.bufferSize;

    holder.ConstructOrcBufferArray(param->bufferArray, decodedChunks);
    param->input = NULL;     // input == NULL is the flag to specify the param is multi-buffered.
    param->input_size = totalSize;
}

void OrcStripeArguemnts::DecodePresentStreams()
{
    // resolve the dependency of present streams.

    // find the parent stream in this stripe
    // start from 1, since 0th column never has the parent
    columnArgs[0].valid_parent_id = -1;
    for (int i = 1; i < columnArgs.size(); i++) {
        int parant_id = FindValidParant(i);
        
        columnArgs[i].valid_parent_id = parant_id;
        columnArgs[i].valid_parent_present = (parant_id < 0) ? NULL : columnArgs[parant_id].stream_present;
    }

    KernelParamBitmap paramBitmap;

    // decode the streams.
    for (int i = 0; i < columnArgs.size(); i++) {
        auto& col = columnArgs[i];
        // skip the column if it is not needed.
        // e.g. the entire columns over all stripes has no present stream. It means there is no null data for the column.
        if (col.type->skipPresentation) continue;
        
        if (!col.stream_present) {
            // It is needed to be cared even though current stripe does not have present stream
            // If all values of present stream in the stripe are non-null, stream is omitted.
            orc_byte* top = deviceArray->at(i).stream_present;
            orc_byte* current = top + (stripe_info->startRowIndex >> 3);
            KernelParamBitmap* param = &paramBitmap;

            param->output = current;
            param->input = NULL;
            param->parent = NULL;
            param->input_size = 0;
            param->start_id = (stripe_info->startRowIndex & 3);
            param->output_count = ((stripe_info->numberOfRows + 7) >> 3);
            param->stat = NULL;

            if ( (stripe_info->startRowIndex & 0x07) == 0 ) {
                current = top + ( stripe_info->startRowIndex >> 3);
                int count = ( (stripe_info->numberOfRows +7) >> 3);
                // ToDo: the stream can also be skipped if the stream is the member of union and there is no data for the stream.
                cudaMemset(current, 0xff, count);
            }
            else {
                EXIT("not suported case.");
                // todo: copy from parent stream.
            }
            continue;
        }

        OrcStreamArguemnts* stream = col.stream_present;
        stream->GetKernelParamBitmap(&paramBitmap);
        DecodeCompressedStream(&paramBitmap, stream);
        cuda_booleanRLEbitmapDepends(&paramBitmap);
    }

#if ( _DEBUG || 0 )
    // force sync for now
    CudaFuncCall(cudaGetLastError());
    CudaFuncCall(cudaDeviceSynchronize());
#endif

#if ( _DEBUG && 1 )    // dump present stream
    D_MSG("[%d]: Column status.", stripe_id);
    D_MSG("[column id]: num of rows, (has present, is no present, varid parent id)");

    for (int i = 0; i < columnArgs.size(); i++) {
        auto& col = columnArgs[i];

        D_MSG("[%d]: %d, (%c, %c, %d)", i, stripe_info->numberOfRows,
            (col.stream_present ? 'Y' : '-' ),
            (col.type->skipPresentation ? 'Y' : '-'),
            col.valid_parent_id
            );

//        if (!col.stream_present)continue;
//        cuda_booleanRLE(col.stream_present->target(), col.stream_present->source(), col.stream_present->sourceSize());
    }

#endif

}

void cuda_intRLE_Depends(KernelParamCommon* param, ORCColumnEncodingKind encode) {
    if ( encode == ORCColumnEncodingKind::OrcDirect_V2 || 
         encode == ORCColumnEncodingKind::OrcDictionary_V2
        ) {
        cuda_integerRLEv2_Depends(param);
    }
    else {
        cuda_integerRLEv1_Depends(param);
    }
}

void GetTimestampKernelParam(KernelParamCoversion& param, OrcColumnArguemnts& col, ORCstream* stream, OrcStripeArguemnts* stripeArg)
{
    param.convertType = OrcKernelConvertionType::GdfTimestampUnit_ms;

    param.input     = col.stream_data->target();
    param.secondary = col.stream_second->target();

    param.data_count = col.stream_data->targetCount();

    // adjustClock is used for converting orc timestamp into gdf timestamp (unix epoch timestamp)
    // adjustClock contains epoch and locale difference.
    // orc epoch (2015/01/01)  => unix epoch (1970/01/01) : 1420070400 = 16436 day * 24 * 60 * 60
    param.adjustClock = 1420070400L + stripeArg->GetGMToffset();

    orc_uint64* output_top = reinterpret_cast<orc_uint64*>( stream->stream_gdf_timestamp );
    param.output = reinterpret_cast<void*>(output_top + col.stream_data->targetStartCount() );
}


// Set up conversion parameter. It converts data and length stream into gdf_string stream
void GetStringDirectKernelParam(KernelParamCoversion& param, OrcColumnArguemnts& col, ORCstream* stream, bool is_dictionay=false)
{
    param.convertType = OrcKernelConvertionType::GdfString_direct;

    param.input = col.stream_data->target();
    param.secondary = col.stream_length->target();

    param.data_count = col.stream_length->targetCount();    // use the target count from length stream since data stream is variable length.

    if (is_dictionay) {
        param.output = col.temporary_buffer;
        param.data_count = col.dictionary_size; // make sure that the elemenet count is the dictionary count.

        // null bitmap won't affect to dictionary decoding
        param.null_bitmap = NULL;
        param.start_id = 0;
    }
    else {
        gdf_string* output_top = reinterpret_cast<gdf_string*>(stream->stream_gdf_string);
        param.output = reinterpret_cast<void*>(output_top + col.stream_length->targetStartCount());

        if (col.stream_present) {
            param.null_bitmap = col.stream_present->target();
            param.start_id = col.stream_present->targetStartCount();
        }
        else {
            param.null_bitmap = NULL;
            param.start_id = 0;
        }
    }
}

// Set up conversion parameter. It converts data and gdf_string of dictionary streams into gdf_string stream
void GetStringDictionaryKernelParam(KernelParamCoversion& param, OrcColumnArguemnts& col, ORCstream* stream)
{
    param.convertType = OrcKernelConvertionType::GdfString_dictionary;

    param.input = col.stream_dictionary_data->target();
    param.secondary = col.temporary_buffer; // gdf_string stream of the dictionary

    // use the target count from length stream since data stream is variable length.
    param.data_count = col.stream_dictionary_data->targetCount();

    param.dict_count = col.dictionary_size;

    gdf_string* output_top = reinterpret_cast<gdf_string*>(stream->stream_gdf_string);
    param.output = reinterpret_cast<void*>(output_top + col.stream_length->targetStartCount());

    if (col.stream_present) {
        param.null_bitmap = col.stream_present->target();
        param.start_id = col.stream_present->targetStartCount();
    }
    else {
        param.null_bitmap = NULL;
        param.start_id = 0;
    }

}

void ClearLengthStream(KernelParamCommon* param)
{
    if (param->present) {
        CudaFuncCall(cudaMemset(param->output, 0, param->output_count) );
    }
}

void OrcStripeArguemnts::DecodeDataStreams()
{
    KernelParamCommon param;

    // decode the streams.
    for (int i = 0; i < columnArgs.size(); i++) {
        auto& col = columnArgs[i];

        if (!col.stream_data)continue;  // struct never has data stream.

        if (col.type->skipDataLoad)continue;    // skip the data load if skipDataLoad is true.

        col.stream_data->GetKernelParamCommon(&param);
        DecodeCompressedStream(&param, col.stream_data);

        switch (col.type->kind) {
        case OrcBolean:
            cuda_booleanByteRLEDepends(&param);
            break;
        case OrcByte:   // TinyInt 
            cuda_ByteRLEDepends(&param);
            break;
        case OrcShort:  // SmallInt, Int, and BigInt
        case OrcInt:
        case OrcLong:
            cuda_intRLE_Depends(&param, col.encoding);
            break;
        case OrcFloat:
        case OrcDouble:
            cuda_raw_data_depends(&param);
            break;
        case OrcDecimal:
            param.elementType = OrcElementType::Sint64;
            cuda_base128_varint_Depends(&param);

            col.stream_second->GetKernelParamCommon(&param);
            DecodeCompressedStream(&param, col.stream_second);

            param.elementType = OrcElementType::Sint32;
            cuda_intRLE_Depends(&param, col.encoding);

            break;

        case OrcDate:
            param.elementType = OrcElementType::Sint64;
            param.convertType = OrcKernelConvertionType::GdfDate64;
            cuda_intRLE_Depends(&param, col.encoding);

            break;

        case OrcTimestamp:
            param.elementType = OrcElementType::Sint64;
            param.convertType = OrcKernelConvertionType::GdfConvertNone;
            cuda_intRLE_Depends(&param, col.encoding);

            col.stream_second->GetKernelParamCommon(&param);
            DecodeCompressedStream(&param, col.stream_second);

#if (GDF_ORC_TIMESTAMP_NANO_PRECISION == 8)
            param.elementType = OrcElementType::Uint64;
#elif (GDF_ORC_TIMESTAMP_NANO_PRECISION == 4)
            param.elementType = OrcElementType::Uint32;
#else
#pragma error "Unsupported size for GDF_ORC_TIMESTAMP_NANO_PRECISION"
#endif
            cuda_intRLE_Depends(&param, col.encoding);

            KernelParamCoversion param_convert;
            GetTimestampKernelParam(param_convert, col, & (deviceArray->at(i)), this);

            cuda_convert_depends(&param_convert);
            break;
        case OrcBinary:
            D_ASSERT(   col.encoding != ORCColumnEncodingKind::OrcDictionary 
                     && col.encoding != ORCColumnEncodingKind::OrcDictionary_V2);
        case OrcVarchar:
        case OrcChar:
        case OrcString:
            switch (col.encoding) {
            case ORCColumnEncodingKind::OrcDictionary:
            case ORCColumnEncodingKind::OrcDictionary_V2:
            {
                // Note: the data and dictionary_data streams are swapped at CudaOrcReaderImplProto::ParseStripeStreams
                // set present stream as NULL since present stream only affects to dictionary id stream
                param.present = NULL;
                if (param.input_size > param.output_count) {
                    D_MSG("output_count and input_size are must be same for string direct encoding. [%d, %d]",
                        param.output_count, param.input_size);
                    EXIT("Failed at Streing decoding");
                }
                cuda_raw_data_depends(&param);

                // decode length stream
                col.stream_length->GetKernelParamCommon(&param);
                DecodeCompressedStream(&param, col.stream_length);
                param.elementType = OrcElementType::Uint32;
                param.present = NULL;   // same reason as data stream
                cuda_intRLE_Depends(&param, col.encoding);

                // decode dictionary id stream
                col.stream_dictionary_data->GetKernelParamCommon(&param);
                DecodeCompressedStream(&param, col.stream_dictionary_data);
                param.elementType = OrcElementType::Uint32;
                cuda_intRLE_Depends(&param, col.encoding);

                // col.dictionary_size := the count of dictionary entry for this stripe
                // allocate gdf_string buffer for dictionary
                AllocateTemporaryBufffer(&col.temporary_buffer, sizeof(gdf_string) * col.dictionary_size);

                // convert data and length streams into gdf_string stream.
                KernelParamCoversion param_convert;
                GetStringDirectKernelParam(param_convert, col, &(deviceArray->at(i)), true);
                cuda_convert_depends(&param_convert);

                // decode each rows from dictinary id by col.temporary_buffer. 
                GetStringDictionaryKernelParam(param_convert, col, &(deviceArray->at(i)) );
                cuda_convert_depends(&param_convert);

                break;
            }
            case ORCColumnEncodingKind::OrcDirect:
            case ORCColumnEncodingKind::OrcDirect_V2:
            {
                // set present stream as NULL since present stream only affects to length stream
                if (param.output_count != param.input_size) {
                    if (compKind == OrcCompressionKind::NONE) {
                        D_MSG("output_count and input_size are must be same for string direct encoding. [%d, %d]",
                            param.output_count, param.input_size);
                        EXIT("Failed at Streing decoding");
                    }
                }
                param.present = NULL;
                cuda_raw_data_depends(&param);

                // decode length stream
                col.stream_length->GetKernelParamCommon(&param);
                DecodeCompressedStream(&param, col.stream_length);
                param.elementType = OrcElementType::Uint32;
                ClearLengthStream(&param);
                cuda_intRLE_Depends(&param, col.encoding);


                // convert data and length streams into gdf_string stream.
                KernelParamCoversion param_convert;
                GetStringDirectKernelParam(param_convert, col, &(deviceArray->at(i)));

                cuda_convert_depends(&param_convert);

                break;
            }
            default:
                EXIT("unknown encoding for string.");
                break;
            }
            break;
        case OrcList:
        case OrcMap:
        case OrcStruct:
        case OrcUnion:
        default:
            break;
        }   // switch (col.type->kind) {

#if ( _DEBUG || 1 )
            // force sync for now
        CudaFuncCall(cudaGetLastError());
        CudaFuncCall(cudaDeviceSynchronize());
#endif


    }   // for (int i = 0; i < columnArgs.size(); i++) {

}



