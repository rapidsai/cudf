/*
 *
 *  Copyright (c) 2023, NVIDIA CORPORATION.
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
 * Regex flags setting, closely following cudf::strings::regex_flags.
 *
 * These types can be or'd to combine them. The values are chosen to
 * leave room for future flags and to match the Python flag values.
 */
public enum RegexFlag {
  DEFAULT(0),   // default
  MULTILINE(8), // the '^' and '$' honor new-line characters
  DOTALL(16),   // the '.' matching includes new-line characters
  ASCII(256);   // use only ASCII when matching built-in character classes

  final int nativeId; // Native id, for use with libcudf.
  private RegexFlag(int nativeId) { // Only constant values should be used
    this.nativeId = nativeId;
  }
}
