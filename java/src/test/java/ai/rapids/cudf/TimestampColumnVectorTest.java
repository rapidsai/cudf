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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TimestampColumnVectorTest {
  static final long[] TIMES_MS = {-131968727238L,   //'1965-10-26 14:01:12.762'
      1530705600000L,   //'2018-07-04 12:00:00.000'
      1674631932929L};  //'2023-01-25 07:32:12.929'

  static final long[] TIMES_S = {-131968728L,   //'1965-10-26 14:01:12'
      1530705600L,   //'2018-07-04 12:00:00'
      1674631932L};  //'2023-01-25 07:32:12'

  @Test
  public void getYear() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      ColumnVector result = timestampColumnVector.year();
      result.ensureOnHost();
      assertEquals(1965, result.getShort(0));
      assertEquals(2018, result.getShort(1));
      assertEquals(2023, result.getShort(2));
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S)) {
      ColumnVector result = timestampColumnVector.year();
      result.ensureOnHost();
      assertEquals(1965, result.getShort(0));
      assertEquals(2018, result.getShort(1));
      assertEquals(2023, result.getShort(2));
    }
  }

  @Test
  public void getMonth() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      ColumnVector result = timestampColumnVector.month();
      result.ensureOnHost();
      assertEquals(10, result.getShort(0));
      assertEquals(7, result.getShort(1));
      assertEquals(1, result.getShort(2));
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S)) {
      ColumnVector result = timestampColumnVector.month();
      result.ensureOnHost();
      assertEquals(10, result.getShort(0));
      assertEquals(7, result.getShort(1));
      assertEquals(1, result.getShort(2));
    }
  }

  @Test
  public void getDay() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      try (ColumnVector result = timestampColumnVector.day()) {
        result.ensureOnHost();
        assertEquals(26, result.getShort(0));
        assertEquals(4, result.getShort(1));
        assertEquals(25, result.getShort(2));
      }
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S);
         ColumnVector result = timestampColumnVector.day()) {
      result.ensureOnHost();
      assertEquals(26, result.getShort(0));
      assertEquals(4, result.getShort(1));
      assertEquals(25, result.getShort(2));
    }
  }

  @Test
  public void getHour() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      try (ColumnVector result = timestampColumnVector.hour()) {
        result.ensureOnHost();
        assertEquals(14, result.getShort(0));
        assertEquals(12, result.getShort(1));
        assertEquals(7, result.getShort(2));
      }
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S);
         ColumnVector result = timestampColumnVector.hour()) {
      result.ensureOnHost();
      assertEquals(14, result.getShort(0));
      assertEquals(12, result.getShort(1));
      assertEquals(7, result.getShort(2));
    }
  }

  @Test
  public void getMinute() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      try (ColumnVector result = timestampColumnVector.minute()) {
        result.ensureOnHost();
        assertEquals(1, result.getShort(0));
        assertEquals(0, result.getShort(1));
        assertEquals(32, result.getShort(2));
      }
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S);
         ColumnVector result = timestampColumnVector.minute()) {
      result.ensureOnHost();
      assertEquals(1, result.getShort(0));
      assertEquals(0, result.getShort(1));
      assertEquals(32, result.getShort(2));
    }
  }

  @Test
  public void getSecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS)) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      try (ColumnVector result = timestampColumnVector.second()) {
        result.ensureOnHost();
        assertEquals(12, result.getShort(0));
        assertEquals(0, result.getShort(1));
        assertEquals(12, result.getShort(2));
      }
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S);
         ColumnVector result = timestampColumnVector.second()) {
      result.ensureOnHost();
      assertEquals(12, result.getShort(0));
      assertEquals(0, result.getShort(1));
      assertEquals(12, result.getShort(2));
    }
  }

  @Test
  public void testCastToTimestamp() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector date64ColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS);
         ColumnVector timestampColumnVector = date64ColumnVector.asTimestamp(TimeUnit.SECONDS)) {
      timestampColumnVector.ensureOnHost();
      assertEquals(-131968728L, timestampColumnVector.getLong(0));
      assertEquals(1530705600L, timestampColumnVector.getLong(1));
      assertEquals(1674631932L, timestampColumnVector.getLong(2));
    }
  }
}
