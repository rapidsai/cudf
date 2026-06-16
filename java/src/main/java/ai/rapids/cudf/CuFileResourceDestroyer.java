/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Destroys a cuFile native resource.
 */
interface CuFileResourceDestroyer {
  void destroy(long pointer);
}
