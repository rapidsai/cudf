/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
