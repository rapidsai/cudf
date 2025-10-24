/*
 * SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
