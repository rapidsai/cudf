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

namespace nvbench {
struct state;
}

/**
 * @brief Sets throughput statistics, such as "Elem/s", "GlobalMem BW", and "BWUtil" for the
 * nvbench results summary.
 *
 * This function could be used to work around a known issue that the throughput statistics
 * should be added before the nvbench::state.exec() call, otherwise they will not be printed
 * in the summary. See https://github.com/NVIDIA/nvbench/issues/175 for more details.
 */
void set_throughputs(nvbench::state& state);
