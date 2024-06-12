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
 *  This class is a Helper class to assert there are no non-empty nulls in a ColumnView
 *
 *  The reason for the existence of this class is so that we can turn the asserts on/off when needed
 *  by passing "-da:ai.rapids.cudf.AssertEmptyNulls". We need that behavior because we have tests
 *  that explicitly test with ColumnViews that contain non-empty nulls but more importantly, there
 *  could be cases where an external system may not have a requirement of nulls being empty, so for
 *  us to work with those systems, we can turn off this assert in the field.
 */
public class AssertEmptyNulls {
  public static void assertNullsAreEmpty(ColumnView cv) {
    if (cv.type.isNestedType() || cv.type.hasOffsets()) {
      assert !cv.hasNonEmptyNulls() : "Column has non-empty nulls";
    }
  }
}
