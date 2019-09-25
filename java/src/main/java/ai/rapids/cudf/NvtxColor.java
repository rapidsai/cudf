/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

public enum NvtxColor {
  GREEN(0xff00ff00),
  BLUE(0xff0000ff),
  YELLOW(0xffffff00),
  PURPLE(0xffff00ff),
  CYAN(0xff00ffff),
  RED(0xffff0000),
  WHITE(0xffffffff),
  DARK_GREEN(0xff006600),
  ORANGE(0xffffa500);

  final int colorBits;

  NvtxColor(int colorBits) {
    this.colorBits = colorBits;
  }
}
