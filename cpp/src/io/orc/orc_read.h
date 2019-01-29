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

#ifndef __ORC_READER_HEADER__
#define __ORC_READER_HEADER__

#include "orc_types.h"
#include <vector>
#include <string>

#define ORC_SET_ERROR_AND_RETURN(err_code) if(GDF_ORC_SUCCESS != err_code){status = err_code; return err_code; }
#define ORC_RETURN_IF_ERROR(expr) { status = (expr); if(GDF_ORC_SUCCESS != status ){return status;}; }

//! ORC footer information (postscript and footer)
struct OrcFooterInfo
{
    // from postscript
    long footerLength;                      //< footer data size [bytes], footer is compressed if compressionKind != 0.
    OrcCompressionKind compressionKind;
    int     compressionBlockSize;           //< the maximum size [bytes] of each compression chunk if compressionKind != 0.
    size_t  metadataLength;                 //< length of metadata
    int     versions[2];                    //< file version
    int     writerVersion;                  //< writer's version

    // from footer
    int     headerlength;        //< must be 3
    size_t  contentlength;       //< size of stripes
    size_t  rowindexstride;      //< max rows in a stride
    size_t  user_metadata_size;  //< metadata data size[bytes]
    size_t  statistics_size;     //< statistics count

    long numberofrows;          //< total number of rows
    long types_size;            //< number of types
    long stripes_size;          //< number of stripes
};

//! constant value for each stripe, needed to point a stripe in the file
struct OrcStripeInfo {
    size_t offset;              //< offset from file top
    size_t indexLength;         //< size[bytes] of stripe index 
    size_t dataLength;          //< size[bytes] of stripe data 
    size_t footerLength;        //< size[bytes] of stripe footer
    size_t numberOfRows;        //< number of rows in the stripe

    size_t startRowIndex;       //< start row index of the column for the stride : accumulation of previous numberOfRows
};

//! the information about variable length stream for the stripe
struct OrcVarLengthInfo {
    size_t numberOfRows;        //< number of rows in the stripe
    size_t startRowIndex;       //< start row index of the column for the stride : accumulation of previous numberOfRows
};

//! decoded stream data for each culumn
struct ORCstream {
    orc_byte* stream_present;
    orc_byte* stream_data;

    // below are optional, used for paticular data types.
    union {
        orc_byte* stream_secondary;
        orc_byte* stream_length;
    };
    union {
        orc_byte* stream_dictionary_data;
    };
    union {
        orc_byte* stream_gdf_string;       //< it is used for a string stream
        orc_byte* stream_gdf_timestamp;    //< it is used for a gdf timestamp date stream, since orc timestamp consists of data and secondary streams
    };

};

//! constant type (column) info over all stripes
struct OrcTypeInfo {
    ORCTypeKind kind;
    size_t subtypes_size;               //< count of subtypes (count of the member of the struct or union)
    std::string fieldNames;             //< name of the field
    orc_uint32 nonNullDataCount;        //< non-null data count, if no null data, it should be same as OrcFooterInfo::numberofrows
    orc_uint32 variableLengthCount;     //< The data length if the data is a kind of list including string, varchar, binary, this may be zero if the data is not a kind of list.
    int parent_id;                      //< id of the parent column, -1 if no parent.
    orc_uint32 nullDataCount;           //< count of null data. 0 if no null data.
    int hasNull;                        //< -1: undefined, 0: no null data, 1: has null data,

    bool isVariableLength;              //< true if the data is a kind of list including string, varchar, binary, this may be zero if the data is not a kind of list.
    bool isValidated;                   //< true if all of varLengthInfo have valid values
    OrcElementType elementType;         //< the format of the output (precision for int/float types)

    bool skipPresentation;              //< skip presentation stream allocate and load if true (default: false)
    bool skipDataLoad;                  //< skip data load if true (default: false)

    //! OrcVarLengthInfo for each stripe if the column is a kind of variable length stream.
    std::vector<OrcVarLengthInfo> varLengthInfo; 

    union {
        struct {
            orc_uint32 maximumLength;   //< maximumLength is used only for varchar or char in UTF-8 characters
        };
        struct {    // precision and scale are used only for decimal data type
            orc_uint32 precision;       //< max precision size of decimal
            orc_uint32 scale;           //< max scale size of decimal
        };
    };

    OrcTypeInfo()
        : parent_id(-1), variableLengthCount(-1), isValidated(false), skipPresentation(false)
        , elementType(OrcElementType::None)
        , skipDataLoad(false)
    {};
};

//! optional paramter for ORC reader
struct OrcReaderOption {
    bool convertToGMT;

    OrcReaderOption()
        : convertToGMT(false)
    {};
};

/** ---------------------------------------------------------------------------*
* @brief Base class of ORC reader
* ---------------------------------------------------------------------------**/
class CudaOrcReader {
public:
    CudaOrcReader() : status(GDF_ORC_SUCCESS) {};
    virtual ~CudaOrcReader() {};

    CudaOrcError_t SetOption(const OrcReaderOption* option);

    virtual CudaOrcError_t ReadFromFile(const char* filename) = 0;

public: // these are debugging or inspecting functions
    const OrcFooterInfo* getFooter() { return &footer_info; }
    virtual const ORCstream& getHost(int i) = 0;


protected:
    OrcFooterInfo                   footer_info;
    std::vector<OrcStripeInfo>      stripes;
    std::vector<OrcTypeInfo>        types;
    bool                            is_no_compression;
    bool                            hasMetadata;
    OrcReaderOption                 option;

    CudaOrcError_t                  status;
};



#endif // __ORC_READER_HEADER__
