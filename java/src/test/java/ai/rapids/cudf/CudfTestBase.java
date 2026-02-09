/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class CudfTestBase {
  static final long RMM_POOL_SIZE_DEFAULT = 512 * 1024 * 1024;

  final int rmmAllocationMode;
  final long rmmPoolSize;

  public CudfTestBase() {
    this(RmmAllocationMode.POOL, RMM_POOL_SIZE_DEFAULT);
  }

  public CudfTestBase(int allocationMode, long poolSize) {
    this.rmmAllocationMode = allocationMode;
    this.rmmPoolSize = poolSize;
  }

  @BeforeEach
  void beforeEach() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    if (!Rmm.isInitialized()) {
      Rmm.initialize(rmmAllocationMode, Rmm.logToStderr(), rmmPoolSize);
    }
  }

  @AfterAll
  static void afterAll() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  private static boolean doublesAreEqualWithinPercentage(double expected, double actual, double percentage) {
    // doubleToLongBits will take care of returning true when both operands have same long value
    // including +ve infinity, -ve infinity or NaNs
    if (Double.doubleToLongBits(expected) != Double.doubleToLongBits(actual)) {
      if (expected != 0) {
        return Math.abs((expected - actual) / expected) <= percentage;
      } else {
        return Math.abs(expected - actual) <= percentage;
      }
    } else {
      return true;
    }
  }

  /**
   * Fails if the absolute difference between expected and actual values as a percentage of the expected
   * value is greater than the threshold
   * i.e. Math.abs((expected - actual) / expected) > percentage, if expected != 0
   * else Math.abs(expected - actual) > percentage
   */
  static void assertEqualsWithinPercentage(double expected, double actual, double percentage) {
     assertEqualsWithinPercentage(expected, actual, percentage, "");
  }

  /**
   * Fails if the absolute difference between expected and actual values as a percentage of the expected
   * value is greater than the threshold
   * i.e. Math.abs((expected - actual) / expected) > percentage, if expected != 0
   * else Math.abs(expected - actual) > percentage
   */
  static void assertEqualsWithinPercentage(double expected, double actual, double percentage, String message) {
    if (!doublesAreEqualWithinPercentage(expected, actual, percentage)) {
      String msg = message + " Math.abs(expected - actual)";
      String eq = (expected != 0 ?
                      " / Math.abs(expected) = " + Math.abs((expected - actual) / expected)
                    : " = " + Math.abs(expected - actual));
      fail(msg + eq + " is not <= " + percentage + " expected(" + expected + ") actual(" + actual + ")");
    }
  }
}
