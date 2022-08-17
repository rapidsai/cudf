#include <cinttypes>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace ipc {
inline void check_cu_status(CUresult res)
{
  if (res != CUDA_SUCCESS) {
    char const* msg;
    if (cuGetErrorString(res, &msg) != CUDA_SUCCESS) {
      CUDF_FAIL("Unable to get CU error explanation.");
    }
    throw logic_error(msg);
  }
}

struct ipc_exported_ptr {
  cudaIpcMemHandle_t handle{0};
  int64_t offset{0};
  int64_t size{0};

  void serialize(std::string* p_bytes) const
  {
    std::string& bytes = *p_bytes;
    size_t this_size   = sizeof(handle) + sizeof(offset) + sizeof(size);
    size_t orig_size   = p_bytes->size();

    p_bytes->resize(orig_size + this_size);
    char* ptr = bytes.data() + orig_size;

    std::memcpy(ptr, &handle, sizeof(handle));
    ptr += sizeof(handle);

    std::memcpy(ptr, &offset, sizeof(offset));
    ptr += sizeof(offset);

    std::memcpy(ptr, &size, sizeof(size));
  }

  static uint8_t const* from_buffer(uint8_t const* ptr, ipc_exported_ptr* out)
  {
    ipc_exported_ptr& dptr = *out;
    std::memcpy(&dptr.handle, ptr, sizeof(handle));
    ptr += sizeof(handle);
    std::memcpy(&dptr.offset, ptr, sizeof(offset));
    ptr += sizeof(offset);
    std::memcpy(&dptr.size, ptr, sizeof(size));
    ptr += sizeof(size);

    return ptr;
  }
};

struct ipc_exported_column {
  ipc_exported_ptr data;
  ipc_exported_ptr mask;

  bool has_nulls() const { return mask.size != 0; }

  void serialize(std::string* p_bytes) const
  {
    std::string& bytes = *p_bytes;
    size_t orig_size   = p_bytes->size();
    auto hn = has_nulls();

    bytes.resize(orig_size + sizeof(hn));
    auto ptr = bytes.data() + orig_size;
    std::memcpy(ptr, &hn, sizeof(hn));

    data.serialize(p_bytes);
    if (has_nulls()) { mask.serialize(p_bytes); }
  }

  static uint8_t const* from_buffer(uint8_t const* ptr, ipc_exported_column* out)
  {
    bool hn;
    std::memcpy(&hn, ptr, sizeof(hn));
    ptr += sizeof(hn);
    std::cout << "hn:" << hn << std::endl;

    ipc_exported_column& column = *out;
    ptr = ipc_exported_ptr::from_buffer(ptr, &column.data);
    if (hn) { ptr = ipc_exported_ptr::from_buffer(ptr, &column.mask); }
    return ptr;
  }
};

inline ipc_exported_ptr export_ptr_for_ipc(uint8_t const* ptr, int64_t size)
{
  CUdeviceptr pbase;
  size_t psize;
  check_cu_status(cuMemGetAddressRange(&pbase, &psize, reinterpret_cast<CUdeviceptr>(ptr)));
  auto base = reinterpret_cast<uint8_t const*>(pbase);

  cudaIpcMemHandle_t handle;
  auto non_const = const_cast<void*>(reinterpret_cast<void const*>(ptr));
  CUDF_CUDA_TRY(cudaIpcGetMemHandle(&handle, non_const));

  return ipc_exported_ptr{.handle = handle, .offset = ptr - base, .size = size};
}
}  // namespace ipc
}  // namespace cudf
