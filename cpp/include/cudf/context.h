/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <memory>

namespace cudf {

namespace jit {
struct ProgramCache;
}

class Context {
 private:
  std::unique_ptr<jit::ProgramCache> _program_cache;

 public:
  Context();
  Context(Context const&)            = delete;
  Context& operator=(Context const&) = delete;
  Context(Context&&)                 = delete;
  Context& operator=(Context&&)      = delete;
  ~Context()                         = default;

  jit::ProgramCache& program_cache();
};

namespace detail {
std::unique_ptr<Context>& context_ptr();
}

Context& context();

void initialize();

void deinitialize();

}  // namespace cudf
