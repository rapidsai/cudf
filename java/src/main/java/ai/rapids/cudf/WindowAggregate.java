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
 */
public class WindowAggregate {
  private final int columnIndex; // Index of column to be aggregated.
  private final WindowAggregateOp windowAggregateOp; // Window-based aggregation operation.

  private WindowAggregate(AggregateOp op, int columnIndex, WindowOptions windowOptions) {
    this.columnIndex = columnIndex;
    this.windowAggregateOp = new WindowAggregateOp(op, windowOptions);
  }

  public static WindowAggregate count(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.COUNT_ALL, columnIndex, windowOptions);
  }

  public static WindowAggregate min(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.MIN, columnIndex, windowOptions);
  }

  public static WindowAggregate max(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.MAX, columnIndex, windowOptions);
  }

  public static WindowAggregate sum(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.SUM, columnIndex, windowOptions);
  }

  public static WindowAggregate mean(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.MEAN, columnIndex, windowOptions);
  }

  public static WindowAggregate median(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.MEDIAN, columnIndex, windowOptions);
  }

  public static WindowAggregate row_number(int columnIndex, WindowOptions windowOptions) {
    return new WindowAggregate(AggregateOp.ROW_NUMBER, columnIndex, windowOptions);
  }

  int getColumnIndex() {
    return columnIndex;
  }

  WindowAggregateOp getOp() {
    return windowAggregateOp;
  }
}
