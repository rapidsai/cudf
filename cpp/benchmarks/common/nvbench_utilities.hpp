/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
