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

#ifndef  __ORC_PROTO_HPP__
#define  __ORC_PROTO_HPP__

#include "orc_types.h"
#include "orc_util.hpp"

#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include "orc_gzip_stream.h"

namespace cudf {
namespace orc {

/** ---------------------------------------------------------------------------*
* @brief A helper template class to load compressed/uncompressed protobuffer message
* ---------------------------------------------------------------------------**/
template<class T>
class CompressedProtoLoader {
public:
    CompressedProtoLoader(OrcCompressionKind kind) : compKind(kind) {};
    ~CompressedProtoLoader() {};

    CudaOrcError_t ParseFromArray(const orc_byte* data, int size) {
        bool ret;
        bool isOriginal = true;

        // if compKind != NONE, read chunk header
        if (OrcCompressionKind::NONE != compKind) {
            const chunk_header *ch = (const chunk_header*)data;
            isOriginal = ch->isOriginal();
            //    D_MSG(" chunk length : %d", ch->getSize());

            // remove chunk header from input
            data += 3;
            size -= 3;
            assert(size == ch->getSize());    // must be same
        }

        if (OrcCompressionKind::NONE == compKind || isOriginal) {
            ret = reader.ParseFromArray((void*)data, size);
        }
        else if (OrcCompressionKind::ZLIB == compKind) {
            google::protobuf::io::ArrayInputStream astream((void*)data, size);
            google::protobuf::io::OrcZlibInputStream gstream(&astream, google::protobuf::io::OrcZlibInputStream::ZLIB, 4 * 256 * 1024);

            ret = reader.ParseFromZeroCopyStream(&gstream);
        }
        else {
            D_MSG("*** this compression format is not supported.");
            return GDF_ORC_UNSUPPORTED_COMPRESSION_TYPE;
        }

        if (!ret) {
            D_MSG("*** fail to parse protocol.");
            return GDF_ORC_INVALID_FILE_FORMAT_PROTOBUF_FAILURE;
        }

        return GDF_ORC_SUCCESS;
    }

    //    const T get() { return reader; };
    const T& get() { return reader; };

protected:
    OrcCompressionKind compKind;
    T reader;
};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_PROTO_HPP__
