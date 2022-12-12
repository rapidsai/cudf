/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda.h>
#include <dlfcn.h>
#include <iostream>

#if defined(__GLIBC__) && __GLIBC__ >= 2 && defined(__GLIBC_MINOR__) && __GLIBC_MINOR__ >= 1
namespace {
static int cuInitCount{0};
using init_t = CUresult (*)(unsigned int);
using proc_t = CUresult (*)(const char*,
                            void**,
                            int,
                            cuuint64_t
#if CUDA_VERSION >= 12000
                            ,
                            CUdriverProcAddressQueryResult*
#endif
);
using dlsym_t = void* (*)(void*, const char*);
static init_t original_cuInit{nullptr};
static proc_t original_cuGetProcAddress{nullptr};
static dlsym_t original_dlsym{nullptr};

static __attribute__((constructor)) void init_cuInit_hack()
{
  // Hack hack hack, relies on matching the exact glibc version
  original_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  if (original_dlsym) {
    original_cuGetProcAddress = (proc_t)original_dlsym(RTLD_NEXT, "cuGetProcAddress");
  }
}

extern "C" {
CUresult cuInit(unsigned int flags)
{
  if (!original_cuInit) {
    void* ptr{nullptr};
    CUresult err = original_cuGetProcAddress("cuInit",
                                             &ptr,
                                             CUDA_VERSION,
                                             CU_GET_PROC_ADDRESS_DEFAULT
#if CUDA_VERSION >= 12000
                                             ,
                                             nullptr
#endif
    );
    if (err != CUDA_SUCCESS) { return err; }
    if (ptr) { original_cuInit = (init_t)(ptr); }
  }
  std::cerr << "cuInit has been called " << ++cuInitCount << " times" << std::endl;
  if (original_cuInit) {
    return original_cuInit(flags);
  } else {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
}

CUresult cuGetProcAddress(const char* symbol,
                          void** pfn,
                          int cudaVersion,
                          cuuint64_t flags
#if CUDA_VERSION >= 12000
                          ,
                          CUdriverProcAddressQueryResult* symbolStatus
#endif
)
{
  if (!original_cuGetProcAddress) { return CUDA_ERROR_NOT_SUPPORTED; }
  CUresult err = original_cuGetProcAddress(symbol,
                                           pfn,
                                           cudaVersion,
                                           flags
#if CUDA_VERSION >= 12000
                                           ,
                                           symbolStatus
#endif
  );
  if (std::string{symbol} == "cuInit") {
    original_cuInit = (init_t)(*pfn);
    *pfn            = (void*)cuInit;
  }
  return err;
}

void* dlsym(void* handle, const char* name_)
{
  std::string name{name_};
  if (name == "cuInit") {
    return (void*)cuInit;
  } else if (name == "cuGetProcAddress") {
    return (void*)cuGetProcAddress;
  } else {
    return original_dlsym(handle, name_);
  }
}
}
}  // namespace
#endif
