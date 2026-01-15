# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that thrust functions in the cpp/ directory use the correct exec_policy.

Rules:
- If the return value of a thrust function is used, use rmm::exec_policy
- If the return value is NOT used, use rmm::exec_policy_nosync

This script analyzes the source code context to determine if the return value
is being used, rather than hardcoding which thrust functions return values.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThrustCall:
    """Represents a thrust function call."""

    file_path: str
    line_number: int
    thrust_function: str
    policy_type: str  # "exec_policy" or "exec_policy_nosync"
    returns_value: bool
    context: str  # The surrounding code context


@dataclass
class Violation:
    """Represents a policy violation."""

    file_path: str
    line_number: int
    line_content: str
    thrust_function: str
    current_policy: str
    expected_policy: str
    message: str


def find_matching_paren(content: str, start_pos: int) -> int:
    """
    Find the position of the closing parenthesis that matches the opening one at start_pos.
    Returns -1 if not found.
    """
    if start_pos >= len(content) or content[start_pos] != "(":
        return -1

    depth = 1
    pos = start_pos + 1

    while pos < len(content) and depth > 0:
        char = content[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == '"':
            # Skip string literals
            pos += 1
            while pos < len(content) and content[pos] != '"':
                if content[pos] == "\\":
                    pos += 1  # Skip escaped character
                pos += 1
        elif char == "'":
            # Skip character literals
            pos += 1
            while pos < len(content) and content[pos] != "'":
                if content[pos] == "\\":
                    pos += 1  # Skip escaped character
                pos += 1
        pos += 1

    return pos - 1 if depth == 0 else -1


def get_line_number(content: str, pos: int) -> int:
    """Get the line number (1-based) for a position in the content."""
    return content[:pos].count("\n") + 1


def is_return_value_used(content: str, call_start: int, call_end: int) -> bool:
    """
    Determine if the return value of a thrust call is used.

    Looks at the context before the thrust:: call to determine if the result
    is being assigned, returned, used in an expression, etc.

    Args:
        content: The full file content
        call_start: Position where "thrust::" starts
        call_end: Position after the closing parenthesis of the call

    Returns:
        True if the return value appears to be used, False otherwise
    """
    # Get the content before the thrust call on the same logical statement
    # We need to find the start of the statement

    # Look backwards to find the start of the statement
    # A statement typically starts after: ; { } or at the start of a line after whitespace
    search_start = max(0, call_start - 500)  # Look back up to 500 chars
    before_content = content[search_start:call_start]

    # Find the last statement boundary
    # Look for ; { } that would indicate the start of a new statement
    last_boundary = -1
    for i in range(len(before_content) - 1, -1, -1):
        char = before_content[i]
        if char in ";{}":
            last_boundary = i
            break

    if last_boundary >= 0:
        statement_prefix = before_content[last_boundary + 1 :]
    else:
        statement_prefix = before_content

    # Strip leading whitespace and newlines
    statement_prefix = statement_prefix.lstrip()

    # Also look at what comes after the call
    after_start = call_end + 1
    after_end = min(len(content), after_start + 100)
    after_content = content[after_start:after_end].lstrip()

    # Patterns that indicate the return value IS being used:

    # 1. Assignment: "auto x = thrust::", "var = thrust::", "Type var = thrust::"
    # Look for = before thrust:: (but not == or != or <= or >=)
    assignment_pattern = re.compile(r"(?:^|[^=!<>])=\s*$", re.MULTILINE)
    if assignment_pattern.search(statement_prefix):
        return True

    # 2. Return statement: "return thrust::"
    if re.search(r"\breturn\s*$", statement_prefix):
        return True

    # 3. Used in a function call argument or constructor
    # e.g., "foo(thrust::reduce(...))" or "SomeClass{thrust::reduce(...)}"
    # Check if there's an unclosed ( or { before the thrust call
    open_parens = statement_prefix.count("(") - statement_prefix.count(")")
    open_braces = statement_prefix.count("{") - statement_prefix.count("}")
    if open_parens > 0 or open_braces > 0:
        return True

    # 4. Used in an expression with operators after the call
    # e.g., "thrust::count(...) > 0" or "thrust::reduce(...) + something"
    # Check what comes after the closing paren
    if re.match(r"\s*[+\-*/%<>=!&|^?:]", after_content):
        return True

    # 5. Used with member access: "thrust::find(...)->something" or "thrust::find(...).something"
    if re.match(r"\s*[.\->]", after_content):
        return True

    # 6. Used with subscript: "thrust::find(...)[index]"
    if re.match(r"\s*\[", after_content):
        return True

    # 7. Initialization in declaration: "auto x = thrust::" or "Type x = thrust::"
    # This is covered by pattern 1, but let's also check for direct initialization
    # "auto x{thrust::...}" or "Type x(thrust::...)"
    if re.search(r"\b\w+\s*[({]\s*$", statement_prefix):
        # Check if this looks like a variable declaration
        decl_match = re.search(
            r"\b(?:auto|const\s+auto|[\w:]+)\s+\w+\s*[({]\s*$",
            statement_prefix,
        )
        if decl_match:
            return True

    # 8. Comma expression or sequence: check if followed by comma in a context
    # where the value matters (like in a tuple or initializer list)
    if re.match(r"\s*,", after_content) and (
        open_parens > 0 or open_braces > 0
    ):
        return True

    # 9. Cast expression: "(Type)thrust::..." or "static_cast<Type>(thrust::...)"
    if re.search(
        r"(?:static_cast|dynamic_cast|reinterpret_cast|const_cast)\s*<[^>]*>\s*\(\s*$",
        statement_prefix,
    ):
        return True
    if re.search(r"\)\s*$", statement_prefix) and re.search(
        r"\(\s*\w+\s*\)\s*$", statement_prefix
    ):
        # C-style cast
        return True

    # 10. Ternary operator: "condition ? thrust::... : ..."
    if re.search(r"\?\s*$", statement_prefix):
        return True

    # 11. Check for dereference: "*thrust::find(...)"
    if re.search(r"\*\s*$", statement_prefix):
        return True

    # If none of the above patterns match, the return value is likely not used
    return False


def find_thrust_calls_with_context(
    content: str, file_path: str
) -> list[ThrustCall]:
    """
    Find all thrust function calls with rmm::exec_policy in the content,
    and determine if their return values are used.

    Returns list of ThrustCall objects.
    """
    results = []

    # Pattern to match thrust::function_name(rmm::exec_policy... or rmm::exec_policy_nosync...
    pattern = re.compile(
        r"thrust::(\w+)\s*\(\s*rmm::exec_policy(_nosync)?\s*\(",
    )

    for match in pattern.finditer(content):
        thrust_func = match.group(1)
        is_nosync = match.group(2) is not None
        policy_type = "exec_policy_nosync" if is_nosync else "exec_policy"

        call_start = match.start()
        line_number = get_line_number(content, call_start)

        # Find the opening paren of the thrust call
        thrust_paren_pos = content.find("(", match.start())
        if thrust_paren_pos == -1:
            continue

        # Find the matching closing paren
        call_end = find_matching_paren(content, thrust_paren_pos)
        if call_end == -1:
            # Could not find matching paren, skip this call
            continue

        # Determine if return value is used
        returns_value = is_return_value_used(content, call_start, call_end)

        # Get context (the line containing the call)
        line_start = content.rfind("\n", 0, call_start) + 1
        line_end = content.find("\n", call_start)
        if line_end == -1:
            line_end = len(content)
        context = content[line_start:line_end].strip()

        results.append(
            ThrustCall(
                file_path=file_path,
                line_number=line_number,
                thrust_function=thrust_func,
                policy_type=policy_type,
                returns_value=returns_value,
                context=context,
            )
        )

    return results


def check_file(
    file_path: Path, verbose: bool = False, nosync_only: bool = False
) -> list[Violation]:
    """Check a single file for policy violations.

    Args:
        file_path: Path to the file to check
        verbose: If True, print details about each thrust call
        nosync_only: If True, only report violations where exec_policy_nosync is expected
    """
    violations = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return violations

    thrust_calls = find_thrust_calls_with_context(content, str(file_path))

    for call in thrust_calls:
        if call.returns_value:
            # Return value is used - should use exec_policy (not nosync)
            if call.policy_type == "exec_policy_nosync" and not nosync_only:
                violations.append(
                    Violation(
                        file_path=call.file_path,
                        line_number=call.line_number,
                        line_content=call.context,
                        thrust_function=call.thrust_function,
                        current_policy=call.policy_type,
                        expected_policy="exec_policy",
                        message=f"thrust::{call.thrust_function} return value is used; should use rmm::exec_policy (not nosync)",
                    )
                )
        else:
            # Return value is NOT used - should use exec_policy_nosync
            if call.policy_type == "exec_policy":
                violations.append(
                    Violation(
                        file_path=call.file_path,
                        line_number=call.line_number,
                        line_content=call.context,
                        thrust_function=call.thrust_function,
                        current_policy=call.policy_type,
                        expected_policy="exec_policy_nosync",
                        message=f"thrust::{call.thrust_function} return value is NOT used; should use rmm::exec_policy_nosync",
                    )
                )

        if verbose:
            status = "USED" if call.returns_value else "NOT USED"
            policy_ok = (
                call.returns_value and call.policy_type == "exec_policy"
            ) or (
                not call.returns_value
                and call.policy_type == "exec_policy_nosync"
            )
            check = "✓" if policy_ok else "✗"
            print(
                f"  {check} {call.file_path}:{call.line_number} thrust::{call.thrust_function} "
                f"[{call.policy_type}] return value {status}",
                file=sys.stderr,
            )

    return violations


def find_cpp_files(directory: Path) -> list[Path]:
    """Find all C++ source files in the directory."""
    extensions = {".cpp", ".cu", ".cuh", ".hpp", ".h"}
    cpp_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in extensions:
                cpp_files.append(file_path)

    return sorted(cpp_files)


def main():
    parser = argparse.ArgumentParser(
        description="Verify thrust exec_policy usage in C++ source files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rules:
  - If the return value of a thrust call is USED, use rmm::exec_policy
  - If the return value is NOT used, use rmm::exec_policy_nosync

The script analyzes the source code context to determine if the return value
is being used (assigned, returned, used in expressions, etc.).

Examples:
  %(prog)s                          # Check cpp/ directory
  %(prog)s --path cpp/src           # Check specific directory
  %(prog)s --verbose                # Show all thrust calls and their analysis
  %(prog)s --exclude-tests          # Exclude test files
  %(prog)s --nosync-only            # Only report where nosync should be used
        """,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to check (default: cpp/ relative to script location)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including all thrust calls analyzed",
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude test files from checking",
    )
    parser.add_argument(
        "--nosync-only",
        default=True,
        action="store_false",  # Default to True
        help="Only report violations where exec_policy_nosync is expected (return value not used)",
    )

    args = parser.parse_args()

    # Determine the path to check
    if args.path:
        check_path = args.path
    else:
        # Default to cpp/ relative to the script's location
        script_dir = Path(__file__).parent
        check_path = script_dir.parent  # cpp/scripts -> cpp/

    if not check_path.exists():
        print(f"Error: Path does not exist: {check_path}", file=sys.stderr)
        return 1

    # Handle both files and directories
    if check_path.is_file():
        cpp_files = [check_path]
        if args.verbose:
            print(f"Checking file: {check_path}", file=sys.stderr)
    elif check_path.is_dir():
        if args.verbose:
            print(f"Checking directory: {check_path}", file=sys.stderr)
        cpp_files = find_cpp_files(check_path)
        if args.exclude_tests:
            cpp_files = [f for f in cpp_files if "/tests/" not in str(f)]
    else:
        print(
            f"Error: Path is not a file or directory: {check_path}",
            file=sys.stderr,
        )
        return 1

    if args.verbose:
        print(f"Found {len(cpp_files)} C++ files to check", file=sys.stderr)

    # Check each file
    all_violations: list[Violation] = []
    files_with_violations = 0

    for file_path in cpp_files:
        violations = check_file(
            file_path, verbose=args.verbose, nosync_only=args.nosync_only
        )
        if violations:
            files_with_violations += 1
            all_violations.extend(violations)

    # Report results
    if all_violations:
        print(
            f"\nFound {len(all_violations)} violation(s) in {files_with_violations} file(s):\n"
        )

        # Group by file
        current_file = None
        for v in sorted(
            all_violations, key=lambda x: (x.file_path, x.line_number)
        ):
            if v.file_path != current_file:
                current_file = v.file_path
                print(f"{v.file_path}:")

            print(f"  Line {v.line_number}: {v.message}")
            print(f"    Current:  rmm::{v.current_policy}")
            print(f"    Expected: rmm::{v.expected_policy}")
            print(
                f"    Code: {v.line_content[:100]}{'...' if len(v.line_content) > 100 else ''}"
            )
            print()

        return 1
    else:
        print(f"No violations found in {len(cpp_files)} files.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
