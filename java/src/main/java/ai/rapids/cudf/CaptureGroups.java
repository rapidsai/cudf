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
 * This is the capture groups setting. For processing a regex pattern containing
 * capture groups. These can be used to optimize the generated regex instructions
 * where the capture groups do not require extracting the groups.
 */
public enum CaptureGroups {
  EXTRACT(0), /* capture groups processed normally for extract */
  NON_CAPTURE(1); /* convert all capture groups to non-capture groups */

  private int captureGroups;

  CaptureGroups(int captureGroups) {
    this.captureGroups = captureGroups;
  }

  public int getValue() {
    return captureGroups;
  }
}
