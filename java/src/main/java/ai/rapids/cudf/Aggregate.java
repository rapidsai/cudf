/*
 *
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * Class for all the aggregate functions like count, max etc
 */
public final class Aggregate {
  private final AggregateOp type;
  private final int index;

  private Aggregate(AggregateOp type, int index) {
    this.type = type;
    this.index = index;
  }

  // Include null in count if include_nulls = true
  static Aggregate count(int index, boolean include_nulls) {
    return new Aggregate(include_nulls ?
            AggregateOp.COUNT_ALL :
            AggregateOp.COUNT_VALID,
            index);
  }

  static Aggregate first(int index, boolean includeNulls) {
    return new Aggregate(includeNulls ?
            AggregateOp.FIRST_INCLUDE_NULLS :
            AggregateOp.FIRST_EXCLUDE_NULLS,
            index);
  }

  static Aggregate last(int index, boolean includeNulls) {
    return new Aggregate(includeNulls ?
            AggregateOp.LAST_INCLUDE_NULLS :
            AggregateOp.LAST_EXCLUDE_NULLS,
            index);
  }

  static Aggregate max(int index) {
    return new Aggregate(AggregateOp.MAX, index);
  }

  static Aggregate min(int index) {
    return new Aggregate(AggregateOp.MIN, index);
  }

  static Aggregate mean(int index) {
    return new Aggregate(AggregateOp.MEAN, index);
  }

  static Aggregate sum(int index) {
    return new Aggregate(AggregateOp.SUM, index);
  }

  static Aggregate median(int index) {
    return new Aggregate(AggregateOp.MEDIAN, index);
  }

  // TODO add in quantile

  int getIndex() {
    return index;
  }

  int getNativeId() {
    return type.nativeId;
  }

  AggregateOp getOp() {
    return type;
  }
}
