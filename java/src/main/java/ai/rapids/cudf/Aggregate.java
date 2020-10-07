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
 * @deprecated please use Aggregation.onColumn instead.
 */
@Deprecated
public final class Aggregate extends AggregationOnColumn {
  private Aggregate(AggregationOnColumn toCopy) {
    super(toCopy.wrapped, toCopy.getColumnIndex());
  }

  /**
   * Include null in count if includeNulls = true
   * @deprecated please use Aggregation.count instead.
   */
  @Deprecated
  static Aggregate count(int index, boolean includeNulls) {
    return new Aggregate(Aggregation.count(includeNulls).onColumn(index));
  }

  /**
   * Get the first element in a list (possibly first non-null element)
   * @deprecated please use Aggregation.nth instead
   */
  @Deprecated
  static Aggregate first(int index, boolean includeNulls) {
    return new Aggregate(Aggregate.nth(0, includeNulls).onColumn(index));
  }

  /**
   * Get the last element in a list (possibly last non-null element)
   * @deprecated please use Aggregation.nth instead
   */
  @Deprecated
  static Aggregate last(int index, boolean includeNulls) {
    return new Aggregate(Aggregate.nth(-1, includeNulls).onColumn(index));
  }

  @Deprecated
  static Aggregate max(int index) {
    return new Aggregate(Aggregation.max().onColumn(index));
  }

  @Deprecated
  static Aggregate min(int index) {
    return new Aggregate(Aggregation.min().onColumn(index));
  }

  @Deprecated
  static Aggregate mean(int index) {
    return new Aggregate(Aggregation.mean().onColumn(index));
  }

  @Deprecated
  static Aggregate sum(int index) {
    return new Aggregate(Aggregation.sum().onColumn(index));
  }

  @Deprecated
  static Aggregate median(int index) {
    return new Aggregate(Aggregation.median().onColumn(index));
  }
}
