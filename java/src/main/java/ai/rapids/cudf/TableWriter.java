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
 * Provides an interface for writing out Table information in multiple steps.
 * A TableWriter will be returned from one of various factory functions in Table that
 * let you set the format of the data and its destination.  After that write can be called one or
 * more times.  When you are done writing call close to finish.
 */
public interface TableWriter extends AutoCloseable {
  /**
   * Write out a table.  Note that all columns must be in the same order each time this is called
   * and the format of each table cannot change.
   * @param table what to write out.
   */
  void write(Table table) throws CudfException;

  @Override
  void close() throws CudfException;
}
