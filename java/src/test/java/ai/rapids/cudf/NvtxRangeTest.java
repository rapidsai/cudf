/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

public class NvtxRangeTest {
  @Test
  public void testNvtxStartEndEnclosed() {
    NvtxRange range1 = new NvtxRange(
        "start/end", NvtxColor.RED, NvtxRange.Type.STARTEND);
    NvtxRange range2 = new NvtxRange(
        "enclosed start/end", NvtxColor.BLUE, NvtxRange.Type.STARTEND);
    range2.close();
    range1.close();
  }

  @Test
  public void testNvtxStartEndCloseOutOfOrder() {
    NvtxRange range1 = new NvtxRange(
        "start/end closes first", NvtxColor.RED, NvtxRange.Type.STARTEND);
    NvtxRange range2 = new NvtxRange(
        "start/end closes later", NvtxColor.BLUE, NvtxRange.Type.STARTEND);
    range1.close();
    range2.close();
  }

  @Test
  public void testNvtxPushPop() {
    try(NvtxRange range1 = new NvtxRange("push/pop", NvtxColor.RED)) {
      try(NvtxRange range2 = new NvtxRange("enclosed push/pop", NvtxColor.BLUE)) {
      }
    }
  }

  @Test
  public void testNvtxPushPopEnclosingStartEnd() {
    try(NvtxRange range1 = new NvtxRange("push/pop", NvtxColor.RED)) {
      NvtxRange range2 = new NvtxRange(
          "enclosed start/end", NvtxColor.BLUE, NvtxRange.Type.STARTEND);
      range2.close();
    }
  }

  @Test
  public void testNvtxPushPopAndStartEndCloseOutOfOrder() {
    NvtxRange range2;
    try(NvtxRange range1 = new NvtxRange("push/pop closes first", NvtxColor.RED)) {
      range2 = new NvtxRange(
          "start/end closes later", NvtxColor.BLUE, NvtxRange.Type.STARTEND);
    }
    range2.close();
  }
}
