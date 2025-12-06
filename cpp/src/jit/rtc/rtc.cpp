
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <jit/rtc/rtc.hpp>
#include <nvJitLink.h>
#include <nvrtc.h>

#define CUDF_CONCATENATE_DETAIL(x, y) x##y
#define CUDF_CONCATENATE(x, y)        CUDF_CONCATENATE_DETAIL(x, y)

#define CUDF_DEFER(...) ::cudf::rtc::defer CUDF_CONCATENATE(defer_, __COUNTER__)(__VA_ARGS__)

namespace cudf {
namespace rtc {

template <typename T>
struct defer {
 private:
  T func_;

 public:
  template <typename... Args>
  defer(Args&&... args) : func_{std::forward<Args>(args)...}
  {
  }
  defer(defer const&)            = delete;
  defer& operator=(defer const&) = delete;
  defer(defer&&)                 = delete;
  defer& operator=(defer&&)      = delete;
  ~defer() { func_(); }
};

fragment fragment_t::load(fragment_t::load_params const& params) { CUDF_FUNC_RANGE(); }

fragment fragment_t::compile(fragment_t::compile_params const& params) { CUDF_FUNC_RANGE(); }

blob fragment_t::get_lto_ir() const { CUDF_FUNC_RANGE(); }

blob fragment_t::get_cubin() const { CUDF_FUNC_RANGE(); }

blob module_t::link_as_cubin(const link_params& params) { CUDF_FUNC_RANGE(); }

module module_t::load(blob_view cubin) { CUDF_FUNC_RANGE(); }

module module_t::link(const link_params& params) { CUDF_FUNC_RANGE(); }

function_ref module_t::get_function(char const* name) const { CUDF_FUNC_RANGE(); }

nvJitLinkHandle load_fragment_from(std::span<unsigned char const> source,
                                   char const* name,
                                   std::span<char const*> link_options,
                                   nvJitLinkInputType type)
{
  CUDF_FUNC_RANGE();

  //   auto sm   = get_device_compute_model();
  //   auto arch = std::format("-arch=sm_{}", sm);
  //   const char* link_options[] = {"-lto", arch.c_str()};

  nvJitLinkHandle handle;
  CUDF_EXPECTS(
    nvJitLinkCreate(&handle, std::size(link_options), link_options.data()) == NVJITLINK_SUCCESS,
    "Failed to create nvJitLink handle");
  CUDF_EXPECTS(
    nvJitLinkAddData(handle, type, source.data(), source.size_bytes(), name) == NVJITLINK_SUCCESS,
    "Failed to add LTO fatbin data to nvJitLink handle");
  CUDF_EXPECTS(nvJitLinkComplete(handle) == NVJITLINK_SUCCESS,
               "Failed to complete nvJitLink handle");

  return handle;
}

// [ ] hash function: runtime? driver? rapids version? etc.

// [ ] cudf_jit_header_map{"<intval>": "int x\0", "intval": "int x\0"} + cuda/std -- always
// available? [ ] when should they be hashed?

// [ ] includes
// [ ] defines
// [ ] link flags
// [ ] compile flags
// [ ] compile for all CUDF architectures?
// input: CUDA code as string
// output: LTO IR code
std::vector<unsigned char> compile_fragment(std::span<char const*> headers,
                                            std::span<char const*> include_names,
                                            std::span<char const*> options)
{
  CUDF_FUNC_RANGE();

  // [ ] create hash
  // [ ] serialize options and headers

  //   auto sm            = get_device_compute_model();
  //   auto arch          = std::format("--gpu-architecture=compute_{}", sm);
  //   char const* opts[] = {arch.c_str(), "--dlto", "--relocatable-device-code=true"};

  nvrtcProgram program;
  nvrtcCreateProgram(&program,
                     "",
                     "program",
                     static_cast<int>(headers.size()),
                     headers.data(),
                     include_names.data());
  nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());

  // nvrtcGetProgramLog(nvrtcProgram prog, char *log)
  // nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet)

  size_t lto_ir_size;
  nvrtcGetLTOIRSize(program, &lto_ir_size);

  std::vector<unsigned char> lto_ir;
  lto_ir.resize(lto_ir_size);
  nvrtcGetLTOIR(program, (char*)lto_ir.data());

  // nvrtcGetErrorString()
  // nvrtcGetCUBIN()
  // nvrtcGetCUBINSize()

  return lto_ir;
}

// input: LTO IR fragments
// output: Linked/Finalized CUBIN. Contains kernel(s) from all fragments
std::vector<unsigned char> link_fragments(std::span<std::span<unsigned char const>> fragments,
                                          std::span<int> fragment_types,
                                          std::span<char const*> names,
                                          std::span<char const*> link_options)
{
  CUDF_FUNC_RANGE();

  // [ ] create hash
  // [ ] serialize link options, names, and fragments
  // [ ] span<char const> exact_hash; request from user

  //   auto sm                 = get_device_compute_model();
  //   auto arch               = std::format("--gpu-architecture=compute_{}", sm);
  //   const char* link_options[] = {"-lto", arch.c_str()};

  std::span<unsigned char const> ltoIR1 = fragments[0];
  char const* ltoIR1_name               = names[0];
  std::span<unsigned char const> ltoIR2 = fragments[1];
  char const* ltoIR2_name               = names[1];
  nvJitLinkHandle handle;
  nvJitLinkCreate(&handle, static_cast<uint32_t>(std::size(link_options)), link_options.data());
  nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void*)ltoIR1.data(), ltoIR1.size(), ltoIR1_name);
  nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void*)ltoIR2.data(), ltoIR2.size(), ltoIR2_name);

  // Call to nvJitLinkComplete causes linker to link together the two LTO IR modules, do
  // optimization on the linked LTO IR, and generate cubin from it.
  nvJitLinkComplete(handle);

  // nvJitLinkGetErrorLog()
  // nvJitLinkGetInfoLog()

  // get linked cubin
  size_t cubinSize;
  nvJitLinkGetLinkedCubinSize(handle, &cubinSize);

  std::vector<unsigned char> cubin;
  cubin.resize(cubinSize);
  nvJitLinkGetLinkedCubin(handle, cubin.data());
  nvJitLinkDestroy(&handle);
  return cubin;
}

// input: linked CUBIN
// output: CUmodule with launchable kernels
CUmodule create_module(std::span<unsigned char const> cubin)
{
  CUDF_FUNC_RANGE();

  CUmodule module;
  // cubin is linked, so now load it
  cuModuleLoadData(&module, cubin.data());
  // cuModuleGetFunctionCount(unsigned int *count, CUmodule mod);
  // cuModuleEnumerateFunctions
  // cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
  // cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name)
  // cuLibraryGetKernelCount(unsigned int *count, CUlibrary lib)

  return module;
}

// SIMPLE KEY: from user
// COMPLEX KEY: sha256 of all parameters and blobs involved; + driver + runtime
//

// [ ] All these functions should have wrappers that cache results and request an optional key from
// the user

// [ ] Method to pre-compile library and reuse it across multiple operators; compile_library();
// compile_library_cached()

// [ ] jit_key(key) -> key+driver+CUDA_versions+CUDA_runtime_versions+device_compute_models
void setup_context()
{
  // #include <cudf_lto_library_fatbin_bytes.h>
}

}  // namespace rtc

}  // namespace cudf
