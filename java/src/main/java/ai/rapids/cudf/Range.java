/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */
package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;

import java.util.function.Consumer;

/**
 * Helper utility for creating ranges.
 */
public final class Range {
  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendBytes(byte end) {
    return appendBytes((byte) 0, end, (byte) 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendBytes(byte start, byte end) {
    return appendBytes(start, end, (byte) 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendBytes(byte start, byte end, byte step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (byte i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendShorts(short end) {
    return appendShorts((short) 0, end, (short) 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendShorts(short start, short end) {
    return appendShorts(start, end, (short) 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendShorts(short start, short end,
                                                                  short step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (short i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendInts(int end) {
    return appendInts(0, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendInts(int start, int end) {
    return appendInts(start, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendInts(int start, int end, int step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (int i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendLongs(long start, long end, long step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (long i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendLongs(long end) {
    return appendLongs(0, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendLongs(long start, long end) {
    return appendLongs(start, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendFloats(float start, float end,
                                                                  float step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (float i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendFloats(float end) {
    return appendFloats(0, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendFloats(float start, float end) {
    return appendFloats(start, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @param step  how must to step by.
   * @return the builder for chaining.
   */
  public static final Consumer<Builder> appendDoubles(double start, double end,
                                                                   double step) {
    assert step > 0;
    assert start <= end;
    return (b) -> {
      for (double i = start; i < end; i += step) {
        b.append(i);
      }
    };
  }

  /**
   * Append a range to the builder. 0 inclusive to end exclusive.
   * @param end last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendDoubles(double end) {
    return appendDoubles(0, end, 1);
  }

  /**
   * Append a range to the builder. start inclusive to end exclusive.
   * @param start first entry.
   * @param end   last entry exclusive.
   * @return the consumer.
   */
  public static final Consumer<Builder> appendDoubles(double start, double end) {
    return appendDoubles(start, end, 1);
  }

}
