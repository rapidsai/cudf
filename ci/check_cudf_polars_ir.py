#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Check IR node consistency in cudf_polars.

Verifies that the `do_evaluate` method signatures in IR subclasses

- Are a classmethod
- Accept `*_non_child` positional arguments, followed by
- `*children` positional arguments, followed by
- A keyword-only `context` argument
"""

from __future__ import annotations

import argparse
import ast
import sys
import typing


class ErrorRecord(typing.TypedDict):
    cls: str
    arg: str
    error: str
    lineno: int
    filename: str


def extract_tuple_from_node(node: ast.AST) -> tuple[str, ...] | None:
    """Extract a tuple of strings from an AST node."""
    if isinstance(node, ast.Tuple):
        return tuple(
            str(elt.value)
            for elt in node.elts
            if isinstance(elt, ast.Constant)
        )
    return None


def get_non_child(class_node: ast.ClassDef) -> tuple[str, ...] | None:
    """Get _non_child attribute from a class definition."""
    for item in class_node.body:
        # Handle annotated assignment: _non_child: ClassVar[...] = (...)
        if isinstance(item, ast.AnnAssign) and isinstance(
            item.target, ast.Name
        ):
            if item.target.id == "_non_child" and item.value:
                return extract_tuple_from_node(item.value)
        # Handle regular assignment: _non_child = (...)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "_non_child":
                    return extract_tuple_from_node(item.value)
    return None


def get_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    """
    Get a method node from a class definition.

    Parameters
    ----------
    class_node : ast.ClassDef
        The class definition to search for the method node.
    name : str
        The name of the method to search for.

    Returns
    -------
    ast.FunctionDef | None
        The method node if found, otherwise None.
        Some nodes (e.g. ErrorNode) don't have a do_evaluate method and
        so `None` is a valid return value.
    """
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def get_non_child_args_length(init_node: ast.FunctionDef) -> int | None:
    """
    Get the length of the tuple assigned to self._non_child_args in __init__.
    Returns None if the assignment is not found or is not a tuple.
    """
    for stmt in ast.walk(init_node):
        # Look for assignments: self._non_child_args = (...)
        if isinstance(stmt, ast.Assign):
            # Check if target is self._non_child_args
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "_non_child_args"
                ):
                    # Check if the value is a tuple
                    if isinstance(stmt.value, ast.Tuple):
                        return len(stmt.value.elts)
    return None


def get_do_evaluate_params(method_node: ast.FunctionDef) -> list[str]:
    """
    Get parameter names from do_evaluate method

    This excludes 'cls' and 'self', and keyword-only params like 'context'.
    """
    params = []
    for arg in method_node.args.args:
        if arg.arg not in ("cls", "self"):
            params.append(arg.arg)
    return params


def get_type_annotation_name(annotation: ast.expr | None) -> str | None:
    """Extract the name from a type annotation."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    # Handle complex types like list[str], dict[str, Any], etc.
    # We just want the base type name
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id
    return None


def is_ir_subclass(class_node: ast.ClassDef) -> bool:
    """Check if a class is a subclass of IR."""
    if not class_node.bases:
        return False

    for base in class_node.bases:
        # Handle both direct imports and aliased imports
        if (isinstance(base, ast.Name) and base.id == "IR") or (
            isinstance(base, ast.Attribute) and base.attr == "IR"
        ):
            return True
    return False


def analyze_content(content: str, filename: str) -> list[ErrorRecord]:
    """Analyze the Python file content for IR node consistency."""
    tree = ast.parse(content, filename=filename)

    records: list[ErrorRecord] = []

    # Find all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not is_ir_subclass(node):
                continue

            class_name = node.name

            non_child = get_non_child(node)
            if non_child is None:
                continue

            method_node = get_node(node, "do_evaluate")
            if method_node is None:
                # Some nodes (e.g. ErrorNode) don't have a do_evaluate method
                continue

            do_evaluate_params = get_do_evaluate_params(method_node)

            for i, nc in enumerate(non_child):
                if nc not in do_evaluate_params:
                    records.append(
                        {
                            "cls": class_name,
                            "arg": nc,
                            "error": "Missing",
                            "lineno": method_node.lineno,
                            "filename": filename,
                        }
                    )
                elif do_evaluate_params.index(nc) != i:
                    records.append(
                        {
                            "cls": class_name,
                            "arg": nc,
                            "error": "Wrong position",
                            "lineno": method_node.lineno,
                            "filename": filename,
                        }
                    )

            # Check that all *remaining* args in do_evaluate are 'DataFrame' type
            regular_args = [
                arg
                for arg in method_node.args.args
                if arg.arg not in ("cls", "self")
            ]

            if method_node.args.vararg is not None:
                regular_args.append(method_node.args.vararg)

            # Check args after _non_child parameters
            for arg in regular_args[len(non_child) :]:
                type_name = get_type_annotation_name(arg.annotation)
                if type_name != "DataFrame":
                    records.append(
                        {
                            "cls": class_name,
                            "arg": arg.arg,
                            "error": f"Wrong type annotation '{type_name}' (expected 'DataFrame')",
                            "lineno": method_node.lineno,
                            "filename": filename,
                        }
                    )

            # Check that the only kw-only argument is 'context' with type 'IRExecutionContext'
            kwonly_args = method_node.args.kwonlyargs
            if len(kwonly_args) != 1:
                records.append(
                    {
                        "cls": class_name,
                        "arg": "kwonly",
                        "error": f"Expected 1 keyword-only argument, found {len(kwonly_args)}",
                        "lineno": method_node.lineno,
                        "filename": filename,
                    }
                )
            elif kwonly_args[0].arg != "context":
                records.append(
                    {
                        "cls": class_name,
                        "arg": kwonly_args[0].arg,
                        "error": "Keyword-only argument should be named 'context'",
                        "lineno": method_node.lineno,
                        "filename": filename,
                    }
                )
            else:
                # Check type annotation
                type_name = get_type_annotation_name(kwonly_args[0].annotation)
                if type_name != "IRExecutionContext":
                    records.append(
                        {
                            "cls": class_name,
                            "arg": "context",
                            "error": f"Wrong type annotation '{type_name}' (expected 'IRExecutionContext')",
                            "lineno": method_node.lineno,
                            "filename": filename,
                        }
                    )

            # Check that __init__ assigns self._non_child_args with matching length
            init_node = get_node(node, "__init__")
            if init_node is not None:
                non_child_args_length = get_non_child_args_length(init_node)
                if non_child_args_length is not None:
                    if non_child_args_length != len(non_child):
                        records.append(
                            {
                                "cls": class_name,
                                "arg": "_non_child_args",
                                "error": "Mismatch between 'self._non_child_args' and 'cls._non_child'",
                                "lineno": init_node.lineno,
                                "filename": filename,
                            }
                        )

    return records


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Check IR node do_evaluate signatures match _non_child declarations"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=argparse.FileType("r"),
        help="Path(s) to Python file(s) to check (use '-' for stdin)",
    )

    args = parser.parse_args()

    all_records: list[ErrorRecord] = []

    try:
        for file in args.files:
            content = file.read()
            filename = file.name
            file.close()

            records = analyze_content(content, filename)
            all_records.extend(records)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if all_records:
        print("Found errors in IR node signatures:", end="\n\n")
        for record in all_records:
            filename = record["filename"]
            lineno = record["lineno"]
            class_name = record["cls"]
            error = record["error"]
            arg = record["arg"]
            print(
                f"  {filename}:{lineno}: {class_name}: {error} argument '{arg}'"
            )
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
