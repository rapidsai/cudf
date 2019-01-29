// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: brianolson@google.com (Brian Olson)
//
// This file contains the definition for classes GzipInputStream and
// GzipOutputStream.
//
// GzipInputStream decompresses data from an underlying
// ZeroCopyInputStream and provides the decompressed data as a
// ZeroCopyInputStream.
//
// GzipOutputStream is an ZeroCopyOutputStream that compresses data to
// an underlying ZeroCopyOutputStream.

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


// Author: takobayashi@nvidia.com (Tadahito Kobayashi)
//
// This file is modified from GzipOutputStream which is ZeroCopyOutputStream
// that compresses data to an underlying ZeroCopyOutputStream.
// Since GzipOutputStream has a static inline function and private member data
// which have to be modified or accessed for ORC reader, this class is introduced.


#ifndef GOOGLE_PROTOBUF_IO_GZIP_STREAM_H__
#define GOOGLE_PROTOBUF_IO_GZIP_STREAM_H__

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <zlib.h>
#include <google/protobuf/stubs/port.h>

namespace google {
namespace protobuf {
namespace io {

// A ZeroCopyInputStream that reads compressed data through zlib
class  OrcZlibInputStream : public google::protobuf::io::ZeroCopyInputStream {
public:
  // Format key for constructor
  enum Format {    // this format is not used for OrcZlibInputStream.
    AUTO = 0,// zlib will autodetect gzip header or deflate stream
    GZIP = 1, // GZIP streams have some extra header data for file attributes.
    ZLIB = 2,// Simpler zlib stream format.
  };

  // buffer_size and format may be -1 for default of 64kB and GZIP format
  explicit OrcZlibInputStream(
      ZeroCopyInputStream* sub_stream,
      Format format = AUTO,
      int buffer_size = -1);
  virtual ~OrcZlibInputStream();

  // Return last error message or NULL if no error.
  inline const char* ZlibErrorMessage() const {
    return zcontext_.msg;
  }
  inline int ZlibErrorCode() const {
    return zerror_;
  }

  // implements ZeroCopyInputStream ----------------------------------
  bool Next(const void** data, int* size);
  void BackUp(int count);
  bool Skip(int count);
  int64 ByteCount() const;

 protected:
  Format format_;

  ZeroCopyInputStream* sub_stream_;

  z_stream zcontext_;
  int zerror_;

  void* output_buffer_;
  void* output_position_;
  size_t output_buffer_length_;
  int64 byte_count_;

  int Inflate(int flush);
  void DoNextOutput(const void** data, int* size);

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(OrcZlibInputStream);
};



}  // namespace io
}  // namespace protobuf
}  // namespace google


#endif  // GOOGLE_PROTOBUF_IO_GZIP_STREAM_H__
