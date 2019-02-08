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

#include "orc_file.h"
#include "orc_debug.h"

#ifdef __linux__
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#elif WIN32
#include <stdlib.h>
#else
#error Unsupported OS.
#endif

namespace cudf {
namespace orc {

CudaOrcError_t 
OrcFile::loadFile(orc_byte*  &file_top, size_t &filesize, const char* filename )
{
#ifdef __linux__
    _fd = open(filename, O_RDONLY);
    if (_fd == -1) {
        D_MSG("*** the file is not found: %s\n", filename);
        return GDF_ORC_FILE_NOT_FOUND;
    }

    struct stat sb;
    if (fstat(_fd, &sb) == -1) {    /* To obtain file size */
        D_MSG("*** fstat error, the file is: %s\n", filename);
        return GDF_ORC_INVALID_FILE_FORMAT;
    }

    if (!sb.st_size) {
        D_MSG("*** the file size is zero, the file is: %s\n", filename);
        return GDF_ORC_INVALID_FILE_FORMAT;
    }
    
    _file_top = (orc_byte*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, _fd, 0);
    if (_file_top < 0) {
        D_MSG("*** failed to map file, the file is: %s\n", filename);
        return GDF_ORC_INVALID_FILE_FORMAT;
    }

    file_top = _file_top;
    filesize = sb.st_size;
    _filesize = filesize;

    return GDF_ORC_SUCCESS;

#elif WIN32
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        D_MSG("*** file is not found: %s\n", filename);
        return GDF_ORC_FILE_NOT_FOUND;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // if the file size is 0, return INVALID_FILE_FORMAT
    if (!filesize) {
        fclose(fp);
        return GDF_ORC_INVALID_FILE_FORMAT;
    }
    file_top = (unsigned char*)malloc(filesize);
    if (!fp) {
        D_MSG("*** allocation failed: %d\n", filesize);
        fclose(fp);
        return GDF_ORC_OUT_OF_MEMORY;
    }

    long ret = fread(file_top, 1, filesize, fp);
    if (ret != filesize) {
        D_MSG("*** fread failed: (%d, %d)\n", ret, filesize);
        fclose(fp);
        return GDF_ORC_INVALID_FILE_FORMAT;
    }

    fclose(fp);
    _filesize = filesize;

    return GDF_ORC_SUCCESS;
#endif
}

void OrcFile::release()
{
#ifdef __linux__
    if (_file_top) {
        munmap(_file_top, _filesize);
        _file_top = NULL;
        _filesize = 0;
        close(_fd);
        _fd = -1;
    }

#elif WIN32
    free(_file_top);
    _file_top = NULL;
#endif
}

}   // namespace orc
}   // namespace cudf
