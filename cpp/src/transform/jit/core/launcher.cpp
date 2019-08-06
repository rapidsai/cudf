/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "../core/launcher.h"
#include <jit/parser.h>
#include <jit/types_h_jit.h>
#include <chrono>
#include <cstdint>
#include "../code/code.h"

namespace cudf {
namespace transformation {
namespace jit {

constexpr char prog_name[] = "unary_transform";

const std::vector<std::string> Launcher::compilerFlags{"-std=c++14"};
const std::vector<std::string> Launcher::headersName{cudf_types_h};

/**
 * @brief  Used to provide Jitify with strings that should be used as headers
 *  during JIT compilation.
 *
 * @param filename  file which was requested to include in source
 * @param stream    stream to pass string of the requested header to
 * @return std::istream*
 **/
std::istream* headersCode(std::string filename, std::iostream& stream) {
  return nullptr;
}

Launcher& Launcher::setProgram(std::string prog_file_name, std::string udf,
                               std::string output_type, bool is_ptx) {
  std::string combined_kernel = is_ptx ?
      parse_single_function_ptx(udf, "GENERIC_UNARY_OP", output_type) +
      code::kernel:
      parse_single_function_cuda(udf, "GENERIC_UNARY_OP", output_type) +
      code::kernel;
  program = cacheInstance.getProgram(prog_file_name, combined_kernel.c_str(),
                                     headersName, compilerFlags, headersCode);
}

Launcher::Launcher(const std::string& udf, const std::string& output_type, bool is_ptx)
    : cacheInstance{cudf::jit::cudfJitCache::Instance()} {
  std::string udf_hash_str =
      std::to_string(std::hash<std::string>{}(udf + output_type));
  this->setProgram(prog_name + ("." + udf_hash_str), udf, output_type, is_ptx);
}

Launcher::Launcher(Launcher&& launcher)
    : program{std::move(launcher.program)},
      cacheInstance{cudf::jit::cudfJitCache::Instance()},
      kernel_inst{std::move(launcher.kernel_inst)} {}

gdf_error Launcher::launch(gdf_column* out, const gdf_column* in) {
  getKernel().configure_1d_max_occupancy().launch(out->size, out->data,
                                                  in->data);

  return GDF_SUCCESS;
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
