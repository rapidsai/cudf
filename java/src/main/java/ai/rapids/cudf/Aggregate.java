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

  static Aggregate count(int index) {
    return new Aggregate(AggregateOp.COUNT, index);
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

  int getIndex() {
    return index;
  }

  int getNativeId() {
    return type.nativeId;
  }
}
