/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.cudf;

import java.io.IOException;

public class DataType {

  DType typeId;
  int scale;
   public DataType(DType id) {typeId = id; }

   public DataType(DType id,  int fp_scale) {
     typeId =id;
     scale = fp_scale;
   }
  private static native long makeNativeType(int typeId, int scale);

  public class NativeDataType implements AutoCloseable {
    long nativeType;
    // build native data_type in constructors
    public NativeDataType(DType id) {
      this.nativeType = DataType.makeNativeType(id.nativeId, 0);
    }
    public NativeDataType(DType id, int decimalScale) {
      this.nativeType = DataType.makeNativeType(id.nativeId, decimalScale);;
    }
    public long returnNativeId() {
      return this.nativeType;
    }

    public void close() throws IOException {
      // call native delete method to clean up native data_type
      //deleteNativeType

    }
  }

}
