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

#include "gdf_orc_util.h"
#include "orc_read_impl_proto.h"
#include "orc_read_impl_gdf.h"

// The public interface
gdf_error gdf_read_orc(
    orc_read_arg *arg
)
{
    using namespace cudf::orc;

    // reset arg first
    arg->data = NULL;
    arg->num_cols_out = 0;
    arg->num_rows_out = 0;

    gdf_error ret;
    CudaOrcReaderImplGdf* reader = new CudaOrcReaderImplGdf;
    if (!reader)return GDF_VALIDITY_UNSUPPORTED; // since there is no OOM error code at gdf yet.

    CudaOrcError_t state;
    OrcReaderOption option;
    option.convertToGMT = arg->convertToGMT;

    state = reader->SetOption(&option);
    if (state != GDF_ORC_SUCCESS) {
        goto ORC_END_READ;
    }

    state = reader->ReadFromFile(arg->file_path);
    if (state != GDF_ORC_SUCCESS) {
        goto ORC_END_READ;
    }

    state = reader->GetGdfArg(arg);

ORC_END_READ:
    delete reader;
    return gdf_orc_convertErrorCode(state);
}

namespace cudf {
namespace orc {

CudaOrcReader* gdf_create_orc_reader() {
    return new CudaOrcReaderImplProto;
}

gdf_error gdf_orc_set_column_name(gdf_column* col, const std::string& name)
{
    if(!col)return GDF_C_ERROR;
    int len = name.length() + 1;
    col->col_name = (char *)malloc(sizeof(char) * len);
    memcpy(col->col_name, name.c_str(), len);
    col->col_name[len - 1] = '\0';

    return GDF_SUCCESS;
}

gdf_error gdf_orc_release_column_name(gdf_column* col)
{
    if (col && col->col_name) {
        free(col->col_name);
        col->col_name = NULL;
    }
    return GDF_SUCCESS;
}

gdf_error gdf_orc_convertErrorCode(CudaOrcError_t err)
{
    switch (err) {
    // SUCCESS and UNSUPPORTED_DATA_TYPE can be success 
    case GDF_ORC_SUCCESS:
        return GDF_SUCCESS;
    case GDF_ORC_UNSUPPORTED_DATA_TYPE:
        return GDF_SUCCESS;
//        return GDF_UNSUPPORTED_DTYPE;

    // all of below are error case
    case GDF_ORC_FILE_NOT_FOUND:
    case GDF_ORC_INVALID_FILE_FORMAT:
    case GDF_ORC_INVALID_FILE_FORMAT_PROTOBUF_FAILURE:
    case GDF_ORC_UNSUPPORTED_COMPRESSION_TYPE:
        return GDF_VALIDITY_UNSUPPORTED;

    case GDF_ORC_OUT_OF_MEMORY:
        return GDF_COLUMN_SIZE_TOO_BIG;
    case GDF_ORC_INVALID_API_CALL:
        return GDF_INVALID_API_CALL;

    case GDF_ORC_MAX_ERROR_CODE:
    default:
        assert("invalid error number.");
        return gdf_error (-1);
    }

}

gdf_dtype gdf_orc_convertDataKind(ORCTypeKind kind)
{
    switch (kind) {
    case OrcByte:
        return GDF_INT8;
    case OrcShort:
        return GDF_INT16;
    case OrcInt:
        return GDF_INT32;
    case OrcLong:
        return GDF_INT64;
    case OrcFloat:
        return GDF_FLOAT32;
    case OrcDouble:
        return GDF_FLOAT64;
    case OrcTimestamp:
        return GDF_TIMESTAMP;
    case OrcDate:
        return GDF_DATE64;
    case OrcString:
    case OrcVarchar:
    case OrcChar:
        return GDF_STRING;
    case OrcBinary:
        return GDF_STRING;
    case OrcBolean:
        return GDF_INT8;
    case OrcDecimal:
    case OrcList:
    case OrcMap:
    case OrcStruct:
    case OrcUnion:
    default:
        return GDF_invalid;
    }


}

}   // namespace orc
}   // namespace cudf

