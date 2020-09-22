/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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
 * Spec for window-based aggregation (for analytical functions)
 * @deprecated use Aggregation.onColumn.overWindow instead.
 */
@Deprecated
public class WindowAggregate extends AggregationOverWindow {
  private WindowAggregate(AggregationOverWindow other) {
    super(other.wrapped, other.columnIndex, other.windowOptions);
  }

  /**
   * @deprecated please use Aggregation.count().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate count(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.count(true)
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.min().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate min(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.min()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.max().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate max(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.max()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.sum().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate sum(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.sum()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.mean().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate mean(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.mean()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.median().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate median(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.median()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }

  /**
   * @deprecated please use Aggregation.rowNumber().onColumn().overWindow()
   */
  @Deprecated
  public static WindowAggregate row_number(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(Aggregation.rowNumber()
            .onColumn(columnIndex)
            .overWindow(windowOptions));
  }
}
