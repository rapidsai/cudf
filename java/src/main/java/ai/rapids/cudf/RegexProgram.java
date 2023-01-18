/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * This is the regex program class.
 */
public class RegexProgram {
  private String pattern; /* regex pattern */
  private RegexFlags flags; /* regex flags for interpreting special characters in the pattern */
  private CaptureGroups capture; /* controls how capture groups in the pattern are used */

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   */
  public RegexProgram(String pattern) {
    this(pattern, RegexFlags.DEFAULT, CaptureGroups.EXTRACT);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   * @param flags Regex flags
   */
  public RegexProgram(String pattern, RegexFlags flags) {
    this(pattern, flags, CaptureGroups.EXTRACT);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   * @param capture Capture groups
   */
  public RegexProgram(String pattern, CaptureGroups capture) {
    this(pattern, RegexFlags.DEFAULT, capture);
  }

  /**
   * Constructor for RegexProgram
   *
   * @param pattern Regex pattern
   * @param flags Regex flags
   * @param capture Capture groups
   */
  public RegexProgram(String pattern, RegexFlags flags, CaptureGroups capture) {
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
   * Get the regex flags used to create this instance
   *
   * @param return Regex flags setting
   */
  public RegexFlags flags() {
    return flags;
  }

  /**
   * Get the capture groups used to create this instance
   *
   * @param return Capture groups setting
   */
  public CaptureGroups capture() {
    return capture;
  }
}
