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

//Evolving and Unstable

/**
 * Interface that allows access only methods to cudf column_views
 */
interface ColumnViewPointerAccess<T> {

  long getColumnView();

  ColumnViewPointerAccess<T> getChildColumnView(int childIndex);

  T getDataBuffer();

  T getOffsetBuffer();

  T getValidityBuffer();

  long getNullCount();

  DType getDataType();

  long getNumRows();

  int getNumChildren();
}
