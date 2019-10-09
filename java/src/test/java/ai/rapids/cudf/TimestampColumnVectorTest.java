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

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TimestampColumnVectorTest {
  static final long[] TIMES_S = {-131968728L,   //'1965-10-26 14:01:12'
                                 1530705600L,   //'2018-07-04 12:00:00'
                                 1674631932L,   //'2023-01-25 07:32:12'
                                 -131968728L,   //'1965-10-26 14:01:12'
                                 1530705600L};  //'2018-07-04 12:00:00'

  static final long[] TIMES_MS = {-131968727762L,   //'1965-10-26 14:01:12.238'
                                  1530705600115L,   //'2018-07-04 12:00:00.115'
                                  1674631932929L,   //'2023-01-25 07:32:12.929'
                                  -131968727762L,   //'1965-10-26 14:01:12.238'
                                  1530705600115L};  //'2018-07-04 12:00:00.115'

  static final long[] TIMES_US = {-131968727761703L,   //'1965-10-26 14:01:12.238297'
                                  1530705600115254L,   //'2018-07-04 12:00:00.115254'
                                  1674631932929861L,   //'2023-01-25 07:32:12.929861'
                                  -131968727761703L,   //'1965-10-26 14:01:12.238297'
                                  1530705600115254L};  //'2018-07-04 12:00:00.115254'

  static final long[] TIMES_NS = {-131968727761702469L,   //'1965-10-26 14:01:12.238297531'
                                  1530705600115254330L,   //'2018-07-04 12:00:00.115254330'
                                  1674631932929861604L,   //'2023-01-25 07:32:12.929861604'
                                  -131968727761702469L,   //'1965-10-26 14:01:12.238297531'
                                  1530705600115254330L};  //'2018-07-04 12:00:00.115254330'

  static final String[] TIMES_S_STRING = {"1965-10-26 14:01:12",
                                          "2018-07-04 12:00:00",
                                          "2023-01-25 07:32:12",
                                          "1965-10-26 14:01:12",
                                          "2018-07-04 12:00:00"};

  static final String[] TIMES_MS_STRING = {"1965-10-26 14:01:12.238000000",
                                           "2018-07-04 12:00:00.115000000",
                                           "2023-01-25 07:32:12.929000000",
                                           "1965-10-26 14:01:12.238000000",
                                           "2018-07-04 12:00:00.115000000"};

  static final String[] TIMES_US_STRING = {"1965-10-26 14:01:12.238297000",
                                           "2018-07-04 12:00:00.115254000",
                                           "2023-01-25 07:32:12.929861000",
                                           "1965-10-26 14:01:12.238297000",
                                           "2018-07-04 12:00:00.115254000"};

  static final String[] TIMES_NS_STRING = {"1965-10-26 14:01:12.238297531",
                                           "2018-07-04 12:00:00.115254330",
                                           "2023-01-25 07:32:12.929861604",
                                           "1965-10-26 14:01:12.238297531",
                                           "2018-07-04 12:00:00.115254330"};

  static final long[] THOUSAND = {1000L, 1000L, 1000L, 1000L, 1000L};

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
        TIMES_S);
      ColumnVector result = timestampColumnVector.year()) {
      result.ensureOnHost();
      assertEquals(1965, result.getShort(0));
      assertEquals(2018, result.getShort(1));
      assertEquals(2023, result.getShort(2));
    }
  }

  @Test
  public void getMonth() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TIMES_MS);
         ColumnVector result = timestampColumnVector.month()) {
      assert timestampColumnVector.getTimeUnit() == TimeUnit.MILLISECONDS;
      result.ensureOnHost();
      assertEquals(10, result.getShort(0));
      assertEquals(7, result.getShort(1));
      assertEquals(1, result.getShort(2));
    }

    try (ColumnVector timestampColumnVector = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS,
        TIMES_S);
         ColumnVector result = timestampColumnVector.month()) {
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

  @Test
  public void testTimestampToLongSecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector ms_string_times = ColumnVector.fromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.fromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector s_expected = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS, TIMES_S);
         ColumnVector s_result = s_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S");
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(s_expected, s_result);
      assertColumnsAreEqual(s_expected, ms_result);
      assertColumnsAreEqual(s_expected, us_result);
      assertColumnsAreEqual(s_expected, ns_result);
    }
  }

  @Test
  public void testTimestampToLongMillisecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector ms_string_times = ColumnVector.fromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.fromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector THOU = ColumnVector.fromLongs(THOUSAND);
         ColumnVector s_expected = ColumnVector.fromLongs(TIMES_S).mul(THOU).asTimestamp(TimeUnit.MILLISECONDS);
         ColumnVector ms_expected = ColumnVector.timestampsFromLongs(TimeUnit.MILLISECONDS, TIMES_MS);
         ColumnVector s_result = s_string_times.asTimestamp(TimeUnit.NONE, "%Y-%m-%d %H:%M:%S");
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.MILLISECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.NONE, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.MILLISECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(s_expected, s_result);
      assertColumnsAreEqual(ms_expected, ms_result);
      assertColumnsAreEqual(ms_expected, us_result);
      assertColumnsAreEqual(ms_expected, ns_result);
    }
  }

  @Test
  public void testTimestampToLongMicrosecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector ms_string_times = ColumnVector.fromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.fromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector THOU = ColumnVector.fromLongs(THOUSAND);
         ColumnVector s_expected = ColumnVector.fromLongs(TIMES_S).mul(THOU).mul(THOU).asTimestamp(TimeUnit.MICROSECONDS);
         ColumnVector ms_expected = ColumnVector.fromLongs(TIMES_MS).mul(THOU).asTimestamp(TimeUnit.MICROSECONDS);
         ColumnVector us_expected = ColumnVector.timestampsFromLongs(TimeUnit.MICROSECONDS, TIMES_US);
         ColumnVector s_result = s_string_times.asTimestamp(TimeUnit.MICROSECONDS, "%Y-%m-%d %H:%M:%S");
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.MICROSECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.MICROSECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.MICROSECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(s_expected, s_result);
      assertColumnsAreEqual(ms_expected, ms_result);
      assertColumnsAreEqual(us_expected, us_result);
      assertColumnsAreEqual(us_expected, ns_result);
    }
  }

  @Test
  public void testTimestampToLongNanosecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector s_string_times = ColumnVector.fromStrings(TIMES_S_STRING);
         ColumnVector ms_string_times = ColumnVector.fromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.fromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.fromStrings(TIMES_NS_STRING);
         ColumnVector THOU = ColumnVector.fromLongs(THOUSAND);
         ColumnVector s_expected = ColumnVector.fromLongs(TIMES_S).mul(THOU).mul(THOU).mul(THOU).asTimestamp(TimeUnit.NANOSECONDS);
         ColumnVector ms_expected = ColumnVector.fromLongs(TIMES_MS).mul(THOU).mul(THOU).asTimestamp(TimeUnit.NANOSECONDS);
         ColumnVector us_expected = ColumnVector.fromLongs(TIMES_US).mul(THOU).asTimestamp(TimeUnit.NANOSECONDS);
         ColumnVector ns_expected = ColumnVector.timestampsFromLongs(TimeUnit.NANOSECONDS, TIMES_NS);
         ColumnVector s_result = s_string_times.asTimestamp(TimeUnit.NANOSECONDS, "%Y-%m-%d %H:%M:%S");
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.NANOSECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.NANOSECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.NANOSECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(s_expected, s_result);
      assertColumnsAreEqual(ms_expected, ms_result);
      assertColumnsAreEqual(us_expected, us_result);
      assertColumnsAreEqual(ns_expected, ns_result);
    }
  }

  @Test
  public void testCategoryTimestampToLongSecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector s_string_times = ColumnVector.categoryFromStrings(TIMES_S_STRING);
         ColumnVector ms_string_times = ColumnVector.categoryFromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.categoryFromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.categoryFromStrings(TIMES_NS_STRING);
         ColumnVector s_expected = ColumnVector.timestampsFromLongs(TimeUnit.SECONDS, TIMES_S);
         ColumnVector s_result = s_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S");
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.SECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(s_expected, s_result);
      assertColumnsAreEqual(s_expected, ms_result);
      assertColumnsAreEqual(s_expected, us_result);
      assertColumnsAreEqual(s_expected, ns_result);
    }
  }

  @Test
  public void testCategoryTimestampToSubsecond() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());

    try (ColumnVector ms_string_times = ColumnVector.categoryFromStrings(TIMES_MS_STRING);
         ColumnVector us_string_times = ColumnVector.categoryFromStrings(TIMES_US_STRING);
         ColumnVector ns_string_times = ColumnVector.categoryFromStrings(TIMES_NS_STRING);
         ColumnVector ms_expected = ColumnVector.timestampsFromLongs(TimeUnit.MILLISECONDS, TIMES_MS);
         ColumnVector us_expected = ColumnVector.timestampsFromLongs(TimeUnit.MICROSECONDS, TIMES_US);
         ColumnVector ns_expected = ColumnVector.timestampsFromLongs(TimeUnit.NANOSECONDS, TIMES_NS);
         ColumnVector ms_result = ms_string_times.asTimestamp(TimeUnit.MILLISECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ms_result_null = ms_string_times.asTimestamp(TimeUnit.NONE, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector us_result = us_string_times.asTimestamp(TimeUnit.MICROSECONDS, "%Y-%m-%d %H:%M:%S.%f");
         ColumnVector ns_result = ns_string_times.asTimestamp(TimeUnit.NANOSECONDS, "%Y-%m-%d %H:%M:%S.%f")) {
      assertColumnsAreEqual(ms_expected, ms_result_null);
      assertColumnsAreEqual(ms_expected, ms_result);
      assertColumnsAreEqual(us_expected, us_result);
      assertColumnsAreEqual(ns_expected, ns_result);
    }
  }
}
