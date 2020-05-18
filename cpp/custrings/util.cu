/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <cstdlib>

#include "nvstrings/NVStrings.h"

#include <cudf/utilities/error.hpp>
#include "./custring_view.cuh"
#include "./util.h"

//
custring_view* custring_from_host(const char* str)
{
  if (!str) return nullptr;
  unsigned int length  = (unsigned int)strlen(str);
  unsigned int bytes   = custring_view::alloc_size(str, length);
  custring_view* d_str = reinterpret_cast<custring_view*>(device_alloc<char>(bytes, 0));
  custring_view::create_from_host(d_str, str, length);
  return d_str;
}

// this is just a convenience and should be removed in the future
NVStrings* createFromCSV(std::string csvfile,
                         unsigned int column,
                         unsigned int lines,
                         unsigned int flags)
{
  FILE* fp = fopen(csvfile.c_str(), "rb");
  if (!fp) {
    printf("Could not open csv file: [%s]\n", csvfile.c_str());
    return nullptr;
  }
  fseek(fp, 0, SEEK_END);
  size_t fileSize = (size_t)ftell(fp);
  fseek(fp, 0, SEEK_SET);
  // printf("File size = %lu bytes\n", fileSize);
  if (fileSize < 2) {
    fclose(fp);
    return nullptr;
  }
  // load file into memory
  size_t contentsSize     = fileSize + 2;
  unsigned char* contents = new unsigned char[contentsSize];
  fileSize                = fread(contents, 1, fileSize, fp);
  contents[fileSize]      = '\r';  // line terminate
  contents[fileSize + 1]  = 0;     // and null-terminate
  fclose(fp);

  // find lines -- compute offsets vector values
  thrust::host_vector<size_t> lineOffsets;
  unsigned char* ptr = contents;
  while (*ptr) {
    unsigned char ch = *ptr;
    if (ch == '\r' || ch == '\n') {
      *ptr = 0;                                // null terminate the line too
      while (ch && (ch < ' ')) ch = *(++ptr);  // skipping any duplicate newline or carriage returns
      lineOffsets.push_back((size_t)(ptr - contents));
      if (lines && (lineOffsets.size() > lines)) break;
      continue;
    }
    ++ptr;
  }
  //
  unsigned int linesCount = (unsigned int)lineOffsets.size();

  // copy file contents into device memory
  char* d_contents = 0;
  CUDA_TRY(cudaMalloc(&d_contents, contentsSize));
  CUDA_TRY(cudaMemcpy(d_contents, contents, contentsSize, cudaMemcpyHostToDevice));
  delete contents;  // done with the host data

  // copy offsets vector into device memory
  thrust::device_vector<size_t> offsets(lineOffsets);
  size_t* d_offsets = offsets.data().get();
  --linesCount;  // header line is skipped
  // printf("Processing %u lines\n",linesCount);
  // build empty output vector of string ptrs
  std::pair<const char*, size_t>* d_index = 0;
  CUDA_TRY(cudaMalloc(&d_index, linesCount * sizeof(std::pair<const char*, size_t>)));
  // create an array of <char*,size_t> pairs pointing to the strings in device memory
  thrust::for_each_n(thrust::device,
                     thrust::make_counting_iterator<size_t>(0),
                     (size_t)linesCount,
                     [d_contents, flags, d_offsets, column, d_index] __device__(size_t idx) {
                       //
                       size_t lineOffset   = d_offsets[idx];
                       size_t lineLength   = d_offsets[idx + 1] - lineOffset;
                       d_index[idx].first  = (const char*)0;
                       d_index[idx].second = 0;
                       if (lineLength < 1) return;
                       char* line = d_contents + lineOffset;
                       char* sptr = line;
                       int length = 0, col = 0;
                       bool bquote = false;  // handle nested quotes
                       for (int i = 0; i < lineLength; ++i) {
                         char ch = line[i];
                         if (ch) {
                           if (ch == '\"') {
                             if (bquote) {
                               if (line[i + 1] != '\"')
                                 bquote = false;
                               else {
                                 ++i;
                                 length += 2;
                               }
                               continue;
                             }
                             if (length == 0) {
                               bquote = true;
                               ++sptr;
                             }
                             continue;
                           }
                           if (bquote || ch != ',') {
                             ++length;
                             continue;
                           }
                         }
                         if (col++ >= column) break;
                         sptr   = line + i + 1;
                         length = 0;
                         bquote = false;
                       }

                       // add string info to array
                       if (length == 0 && ((flags & CSV_NULL_IS_EMPTY) == 0))
                         d_index[idx].first = 0;
                       else
                         d_index[idx].first = (const char*)sptr;
                       d_index[idx].second = (size_t)length;
                     });

  CUDA_TRY(cudaDeviceSynchronize());
  // the NVStrings object can now be created from the array of pairs
  NVStrings::sorttype stype = (NVStrings::sorttype)(flags & (CSV_SORT_LENGTH | CSV_SORT_NAME));
  NVStrings* rtn = NVStrings::create_from_index(d_index, (unsigned int)linesCount, true, stype);
  cudaFree(d_index);     // done with string index array
  cudaFree(d_contents);  // done with csv device memory
  return rtn;
}
