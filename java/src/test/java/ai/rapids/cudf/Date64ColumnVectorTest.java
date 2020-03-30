/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class Date64ColumnVectorTest extends CudfTestBase {
  private static final long[] DATES = {-131968727238L,   //'1965-10-26 14:01:12.762'
      1530705600000L,   //'2018-07-04 12:00:00.000'
      1674631932929L};  //'2023-01-25 07:32:12.929'

  @Test
  public void getYear() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.year();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(1965, result.getShort(0));
      assertEquals(2018, result.getShort(1));
      assertEquals(2023, result.getShort(2));
    }
  }

  @Test
  public void getMonth() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.month();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(10, result.getShort(0));
      assertEquals(7, result.getShort(1));
      assertEquals(1, result.getShort(2));
    }
  }

  @Test
  public void getDay() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.day();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(26, result.getShort(0));
      assertEquals(4, result.getShort(1));
      assertEquals(25, result.getShort(2));
    }
  }

  @Test
  public void getHour() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.hour();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(14, result.getShort(0));
      assertEquals(12, result.getShort(1));
      assertEquals(7, result.getShort(2));
    }
  }

  @Test
  public void getMinute() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.minute();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(1, result.getShort(0));
      assertEquals(0, result.getShort(1));
      assertEquals(32, result.getShort(2));
    }
  }

  @Test
  public void getSecond() {
    try (ColumnVector date64ColumnVector = ColumnVector.timestampMilliSecondsFromLongs(DATES);
         ColumnVector tmp = date64ColumnVector.second();
         HostColumnVector result = tmp.copyToHost()) {
      assertEquals(12, result.getShort(0));
      assertEquals(0, result.getShort(1));
      assertEquals(12, result.getShort(2));
    }
  }
}
