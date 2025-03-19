/*
 *
 *  Copyright (c) 2025, NVIDIA CORPORATION.
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
 * Used for the dropListDuplicates function
 * Specifies which duplicate to keep
 * @see cudf::duplicate_keep_option in /cpp/include/cudf/stream_compaction.hpp,
 * from which this enum is based off of. Values should be kept in sync.
 */
public enum DuplicateKeepOption {
  KEEP_ANY(0),    // keep any instance of a value
  KEEP_FIRST(1),  // only keep the first instance of an value
  KEEP_LAST(2),   // keep the last instance of an value
  KEEP_NONE(3);   // remove all instances of values with duplicates
  final int nativeId;

  DuplicateKeepOption(int nativeId) { this.nativeId = nativeId; }
}
