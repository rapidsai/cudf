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

#include "orc_read_impl_gdf.h"
#include "gdf_orc_util.h"

namespace cudf {
namespace orc {

CudaOrcError_t CudaOrcReaderImplGdf::ReadFromFile(const char* filename)
{
    return CudaOrcReaderImplProto::ReadFromFile(filename);
}


void CudaOrcReaderImplGdf::SelectColumns()
{

}

// for now, the data type is very limited support!
CudaOrcError_t CudaOrcReaderImplGdf::GetGdfArg(orc_read_arg *arg)
{
    // pre-requirement: ReadFromFile() returns GDF_ORC_SUCCESS.
    if (status != GDF_ORC_SUCCESS)return status;

    std::vector<gdf_column*> colmns;    // output

    for (int i = 0; i < deviceArray.size(); i++) {
        if (!IsSupportedType(i))continue;

        auto& stream = deviceArray[i];
        auto& type = types[i];

        auto* col = new gdf_column();

        col->size = footer_info.numberofrows;           // if the data is a kind of list series, this will be different from table.num_records.
        col->dtype = gdf_orc_convertDataKind(type.kind);
        col->col_name = NULL;
        col->null_count = type.nullDataCount;
        col->valid = stream.stream_present;
        col->dtype_info.time_unit = TIME_UNIT_NONE;
        stream.stream_present = NULL;                   // detach from CudaOrcReaderImplGdf

        // set the data stream into gdf output, then detach the data stream from orc reader.
        switch (type.kind) {
        case ORCTypeKind::OrcTimestamp:
            col->data = stream.stream_gdf_timestamp;
            stream.stream_gdf_timestamp = NULL; // detach from CudaOrcReaderImplGdf
            break;
        case ORCTypeKind::OrcVarchar:
        case ORCTypeKind::OrcChar:
        case ORCTypeKind::OrcString:
        case ORCTypeKind::OrcBinary:
            col->data = stream.stream_gdf_string;
            stream.stream_gdf_string = NULL;
            stream.stream_data = NULL; // detach data stream because it is backed buffer referenced by stream_gdf_string 
            break;
        default:
            col->data = stream.stream_data;
            stream.stream_data = NULL;
            break;
        }

        gdf_orc_set_column_name(col, type.fieldNames);

        colmns.push_back(col);
    }

    if (!colmns.size())return GDF_ORC_UNSUPPORTED_DATA_TYPE;

    arg->num_cols_out = colmns.size();
    arg->num_rows_out = footer_info.numberofrows;
    arg->data = new gdf_column*[colmns.size()];
    memcpy((void*)arg->data, (void*)colmns.data(), sizeof(gdf_column*) * colmns.size() );

    return status;
}

//! return false if the type is not supported at gdf
bool CudaOrcReaderImplGdf::IsSupportedType(int id)
{
    auto& type = types[id];

    switch (type.kind) {
    case OrcBolean:
    case OrcByte:
    case OrcShort:
    case OrcInt:
    case OrcLong:
    case OrcFloat:
    case OrcDouble:
    case OrcTimestamp:
    case OrcDate:
    case OrcVarchar:
    case OrcChar:
    case OrcBinary:
    case OrcString:
        if (type.parent_id == 0 || type.parent_id == -1)return true;    // nested struct/list/map/union is not supported yet
        break;
    case OrcDecimal:
    case OrcList:
    case OrcMap:
    case OrcStruct:
    case OrcUnion:
    default:
        break;
    }
    

    return false;
}

}   // namespace orc
}   // namespace cudf



