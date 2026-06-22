/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.EnumSet;

/**
 * Regex program class, closely following cudf::strings::regex_program.
 */
public class RegexProgram {
  private String pattern; // regex pattern
  // regex flags for interpreting special characters in the pattern
  private EnumSet<RegexFlag> flags;
  // controls how capture groups in the pattern are used
  // default is to extract a capture group
  private CaptureGroups capture;

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   */
  public RegexProgram(String pattern) {
    this(pattern, EnumSet.of(RegexFlag.DEFAULT), CaptureGroups.EXTRACT);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   * @param flags Regex flags setting
   */
  public RegexProgram(String pattern, EnumSet<RegexFlag> flags) {
    this(pattern, flags, CaptureGroups.EXTRACT);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern setting
   * @param capture Capture groups setting
   */
  public RegexProgram(String pattern, CaptureGroups capture) {
    this(pattern, EnumSet.of(RegexFlag.DEFAULT), capture);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   * @param flags Regex flags setting
   * @param capture Capture groups setting
   */
  public RegexProgram(String pattern, EnumSet<RegexFlag> flags, CaptureGroups capture) {
    assert pattern != null : "pattern may not be null";
    this.pattern = pattern;
    this.flags = flags;
    this.capture = capture;
  }

  /**
   * Get the pattern used to create this instance
   *
   * @param return A regex pattern as a string
   */
  public String pattern() {
    return pattern;
  }

  /**
   * Get the regex flags setting used to create this instance
   *
   * @param return Regex flags setting
   */
  public EnumSet<RegexFlag> flags() {
    return flags;
  }

  /**
   * Reset the regex flags setting for this instance
   *
   * @param flags Regex flags setting
   */
  public void setFlags(EnumSet<RegexFlag> flags) {
    this.flags = flags;
  }

  /**
   * Get the capture groups setting used to create this instance
   *
   * @param return Capture groups setting
   */
  public CaptureGroups capture() {
    return capture;
  }

  /**
   * Reset the capture groups setting for this instance
   *
   * @param capture Capture groups setting
   */
  public void setCapture(CaptureGroups capture) {
    this.capture = capture;
  }

  /**
   * Combine the regex flags using 'or'
   *
   * @param return An integer representing the value of combined (or'ed) flags
   */
  public int combinedFlags() {
    int allFlags = 0;
    for (RegexFlag flag : flags) {
      allFlags |= flag.nativeId;
    }
    return allFlags;
  }
}
