/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#pragma once

#include <nvbench/nvbench.cuh>

class nvbench_criterion final : public nvbench::stopping_criterion_base {
  nvbench::int64_t m_num_samples{};

 public:
  nvbench_criterion()
    : nvbench::stopping_criterion_base{"fixed", {{"max-samples", nvbench::int64_t{20}}}}
  {
  }

 protected:
  // Setup the criterion in the `do_initialize()` method:
  virtual void do_initialize() override { m_num_samples = 0; }

  // Process new measurements in the `add_measurement()` method:
  virtual void do_add_measurement(nvbench::float64_t) override { m_num_samples++; }

  // Check if the stopping criterion is met in the `is_finished()` method:
  virtual bool do_is_finished() override
  {
    return m_num_samples >= m_params.get_int64("max-samples");
  }
};

NVBENCH_REGISTER_CRITERION(nvbench_criterion);
