/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Provides an interface for reading multiple tables from a single input source.
 */
public interface StreamedTableReader extends AutoCloseable {
    /**
     * Get the next table if available.
     * @return the next Table or null if done reading tables.
     * @throws CudfException on any error.
     */
    Table getNextIfAvailable() throws CudfException;

    /**
     * Get the next table if available.
     * @param rowTarget the target number of rows to read (this is really just best effort).
     * @return the next Table or null if done reading tables.
     * @throws CudfException on any error.
     */
    Table getNextIfAvailable(int rowTarget) throws CudfException;

    @Override
    void close() throws CudfException;
}
