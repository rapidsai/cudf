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

public class AssertEmptyNulls {
  // This is the recommended way to check at runtime that assertions are enabled.
  // https://docs.oracle.com/javase/8/docs/technotes/guides/language/assert.html
  private static boolean assertNullsAreEmpty = false;
  static {
      assert assertNullsAreEmpty = true ; // Intentional side effect
  }

  public static void assertNullsAreEmpty(ColumnView cv) {
    if (cv.type.isNestedType() || cv.type.hasOffsets() && assertNullsAreEmpty) {
      assert !cv.hasNonEmptyNulls() : "Column has non-empty nulls";
    }
  }
}
