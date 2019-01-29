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

#ifndef __ORC_READER_PROTOCOL_BUFFER_HEADER__
#define __ORC_READER_PROTOCOL_BUFFER_HEADER__


#include "orc_read.h"
#include "orc_debug.h"
#include "orc_proto.pb.h"    // from google protocol buffers.
#include "orc_util.hpp"
#include "orc_stripe.h"
#include "orc_file.h"

/** ---------------------------------------------------------------------------*
* @brief implementation of reading protocol buffer from ORC file
* ---------------------------------------------------------------------------**/
class CudaOrcReaderImplProto : public CudaOrcReader {
public:
    CudaOrcReaderImplProto() : file_top(NULL){};
    virtual ~CudaOrcReaderImplProto();

    virtual CudaOrcError_t ReadFromFile(const char* filename);

    virtual void SelectColumns();
    
    void release();

public:
    const ORCstream& getHost(int i);

protected:
    void GetPSInfo(const orc::proto::PostScript& ps);
    void GetFooterInfo(const orc::proto::Footer& ft);
    void GetStripesInfo(const orc::proto::Footer& ft);
    void GetTypesInfo(const orc::proto::Footer& ft);
    void GetMetaDataInfo(const orc::proto::Metadata& mt);

    bool isVariableContainer(int id);
    bool isVariableLength(int id);

    bool ValidateColumns();

    orc_uint32 findGMToffset(const char* region);

    void ParseStripeColumns(const orc::proto::StripeFooter& stripe, OrcStripeArguemnts& stripeArg);
    void ParseStripeStreams(const orc::proto::StripeFooter& stripe, OrcStripeArguemnts& stripeArg);

    void Decode();

    CudaOrcError_t AllocateArrays();
    CudaOrcError_t DeAllocateArrays();

protected:  // -- debug features
    void dumpFooterInfo();
    void dumpStripes();
    void dumpTypes();
    void dumpMetaData(const orc::proto::Metadata& mt);
    void dumpStreamInfo(const orc::proto::StripeFooter& ft);

protected:
    OrcFile orc_file;                           //< file loader class
    orc_byte* file_top;                         //< mapped buffer of the ORC file 
    std::map<std::string, int> timezoneCache;   //< timezone cache

    std::vector<ORCstream> deviceArray;         //< streams decoded at GPU. size of arrays is same as size of types.
    std::vector<OrcStripeArguemnts> stripeArgs; //< arguments for each stripe
};


#endif // __ORC_READER_PROTOCOL_BUFFER_HEADER__
