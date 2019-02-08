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

#ifndef __ORC_FILE_HEADER__
#define __ORC_FILE_HEADER__

#include "orc_types.h"

namespace cudf {
namespace orc {


class OrcFile {
public:
    OrcFile() : _file_top(NULL), _filesize(0){};
    ~OrcFile() {release();};

public:
    CudaOrcError_t loadFile(orc_byte*  &file_top, size_t &filesize, const char* filename );

    void release();
    
protected:
    orc_byte*   _file_top;
    size_t      _filesize;

#ifdef __linux__
    int         _fd;
#endif

};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_FILE_HEADER__
