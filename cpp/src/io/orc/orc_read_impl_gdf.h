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

#ifndef __ORC_READER_GDF_HEADER__
#define __ORC_READER_GDF_HEADER__

#include "orc_read_impl_proto.h"

#include <cudf.h>

/** ---------------------------------------------------------------------------------*
* @brief implementation of converting from ORC reader internal format into gdf format
* -----------------------------------------------------------------------------------**/
// implement output gdf format
class CudaOrcReaderImplGdf : public CudaOrcReaderImplProto {
public:
    CudaOrcReaderImplGdf()  {};
    virtual ~CudaOrcReaderImplGdf() {};

    virtual CudaOrcError_t ReadFromFile(const char* filename);
    CudaOrcError_t GetGdfArg(orc_read_arg *arg);

    virtual void SelectColumns();

protected:
    bool IsSupportedType(int id);
};



#endif // __ORC_READER_GDF_HEADER__
