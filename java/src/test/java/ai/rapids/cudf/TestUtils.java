/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static java.lang.Double.NEGATIVE_INFINITY;
import static java.lang.Double.POSITIVE_INFINITY;

/**
 * Utility class for generating test data
 */
class TestUtils {

  static int NULL = 0x00000001;
  static int ZERO = 0x00000002;
  static int INF = 0x00000004;
  static int NAN = 0x00000008;
  static int NEG_ZERO = 0x00000010;
  static int ALL = NULL|ZERO|INF|NAN|NEG_ZERO;
  static int NONE = 0;

  private static boolean hasZero(int v) {
    return (v & ZERO) > 0;
  }

  private static boolean hasNegativeZero(int v) {
    return (v & NEG_ZERO) > 0;
  }

  private static boolean hasNan(int v) {
    return (v & NAN) > 0;
  }

  private static boolean hasNull(int v) {
    return (v & NULL) > 0;
  }

  private static boolean hasInf(int v) {
    return (v & INF) > 0;
  }

  static Long[] getLongs(final long seed, final int size) {
    return getLongs(seed, size, ALL);
  }

  static Double[] getDoubles(final long seed, final int size) {
    return getDoubles(seed, size, ALL);
  }

  static Integer[] getIntegers(final long seed, final int size) {
    return getIntegers(seed, size, ALL);
  }

  /**
   * A convenience method for generating a fixed set of Integer values. This is by no means uniformly
   * distributed. i.e. some values have more probability of occurrence than others.
   *
   * @param seed Seed value to be used to generate values
   * @param size Number of values to be generated
   * @param specialValues Values to include. Please refer to {@link TestUtils#ALL} for possible values
   */
  static Long[] getLongs(final long seed, final int size, int specialValues) {
    Random r = new Random(seed);
    Long[] result = new Long[size];
    List<Long> v = new ArrayList();
    if (hasZero(specialValues)) v.add(0L);
    if (hasNull(specialValues)) v.add(null);

    Long[] v_arr = v.stream().toArray(Long[]::new);

    IntStream.range(0, size).forEach(index -> {
      switch (r.nextInt(v_arr.length + 4)) {
        case 0:
          result[index] = (long) (Long.MAX_VALUE * r.nextDouble());
          break;
        case 1:
          result[index] = (long) (Long.MIN_VALUE * r.nextDouble());
          break;
        case 2:
          result[index] = Long.MIN_VALUE;
          break;
        case 3:
          result[index] = Long.MAX_VALUE;
          break;
        case 4:
          result[index] = v_arr[0];
          break;
        default:
          result[index] = v_arr[1];
      }
    });
    return result;
  }

  /**
   * A convenience method for generating a fixed set of Integer values. This is by no means uniformly
   * distributed. i.e. some values have more probability of occurrence than others.
   *
   * @param seed Seed value to be used to generate values
   * @param size Number of values to be generated
   * @param specialValues Values to include. Please refer to {@link TestUtils#ALL} for possible values
   */
  static Integer[] getIntegers(final long seed, final int size, int specialValues) {
    Random r = new Random(seed);
    Integer[] result = new Integer[size];
    List<Integer> v = new ArrayList();
    if (hasZero(specialValues)) v.add(0);
    if (hasNull(specialValues)) v.add(null);

    Integer[] v_arr = v.stream().toArray(Integer[]::new);

    IntStream.range(0, size).forEach(index -> {
      switch (r.nextInt(v_arr.length + 4)) {
        case 0:
          result[index] = (int) (Integer.MAX_VALUE * r.nextDouble());
          break;
        case 1:
          result[index] = (int) (Integer.MIN_VALUE * r.nextDouble());
          break;
        case 2:
          result[index] = Integer.MIN_VALUE;
          break;
        case 3:
          result[index] = Integer.MAX_VALUE;
          break;
        case 4:
          result[index] = v_arr[0];
          break;
        default:
          result[index] = v_arr[1];
      }
    });
    return result;
  }

  /**
   * A convenience method for generating a fixed set of Double values. This is by no means uniformly
   * distributed. i.e. some values have more probability of occurrence than others.
   *
   * @param seed Seed value to be used to generate values
   * @param size Number of values to be generated
   * @param specialValues Values to include. Please refer to {@link TestUtils#ALL} for possible values
   */
  static Double[] getDoubles(final long seed, final int size, int specialValues) {
    Random r = new Random(seed);
    Double[] result = new Double[size];
    List<Double> v = new ArrayList();
    if (hasZero(specialValues)) v.add(0.0);
    if (hasNegativeZero(specialValues)) v.add(-0.0);
    if (hasInf(specialValues)) {
      v.add(POSITIVE_INFINITY);
      v.add(NEGATIVE_INFINITY);
    }
    if (hasNan(specialValues)) v.add(Double.NaN);
    if (hasNull(specialValues)) v.add(null);

    Double[] v_arr = v.stream().toArray(Double[]::new);

    IntStream.range(0, size).forEach(index -> {
      switch (r.nextInt(v_arr.length + 4)) {
        case 0:
          result[index] = 1 + (Double.MAX_VALUE * r.nextDouble() - 2);
          break;
        case 1:
          result[index] = 1 + (Double.MIN_VALUE * r.nextDouble() - 2);
          break;
        case 2:
          result[index] = Double.MIN_VALUE;
          break;
        case 3:
          result[index] = Double.MAX_VALUE;
          break;
        case 4:
          result[index] = v_arr[0];
          break;
        case 5:
          result[index] = v_arr[1];
          break;
        case 6:
          result[index] = v_arr[2];
          break;
        case 7:
          result[index] = v_arr[3];
          break;
        case 8:
          result[index] = v_arr[4];
          break;
        default:
          result[index] = v_arr[5];
      }
    });
    return result;
  }

  public static File getResourceAsFile(String resourceName) {
    URL url = TestUtils.class.getClassLoader().getResource(resourceName);
    if (url == null) {
      throw new IllegalArgumentException("Unable to locate resource: " + resourceName);
    }
    try {
      return new File(url.toURI());
    } catch (URISyntaxException e) {
      throw new RuntimeException(e);
    }
  }
}
