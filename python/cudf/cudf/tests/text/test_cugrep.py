import os
import sys
import pytest
from cudf.core.tools.text import cugrep

def test_cugrep_basic(tmp_path):
    # Create a sample CSV file
    csv_content = """col1,col2\nhello world,foo\nbar baz,qux\nhello cudf,bar\n"""
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    # Run cugrep on the file
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cugrep.cugrep, ["hello", str(file_path), "--column", "col1"])
    assert result.exit_code == 0
    assert "hello world" in result.output
    assert "hello cudf" in result.output
    assert "bar baz" not in result.output

def test_cugrep_all_columns(tmp_path):
    csv_content = """col1,col2\nhello world,foo\nbar baz,hello again\nhello cudf,bar\n"""
    file_path = tmp_path / "test2.csv"
    file_path.write_text(csv_content)
    from click.testing import CliRunner
    runner = CliRunner()
    # No --column: should match both columns
    result = runner.invoke(cugrep.cugrep, ["hello", str(file_path)])
    assert result.exit_code == 0
    assert "hello world" in result.output
    assert "hello cudf" in result.output
    assert "hello again" in result.output

def test_cugrep_ignore_case(tmp_path):
    csv_content = "col1\nHello World\nHELLO cudf\nno match\n"
    file_path = tmp_path / "test3.csv"
    file_path.write_text(csv_content)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cugrep.cugrep, ["hello", str(file_path), "--ignore-case", "--column", "col1"])
    assert result.exit_code == 0
    assert "Hello World" in result.output
    assert "HELLO cudf" in result.output
    assert "no match" not in result.output

def test_cugrep_missing_column(tmp_path):
    csv_content = "col1\nfoo\nbar\n"
    file_path = tmp_path / "test4.csv"
    file_path.write_text(csv_content)
    from click.testing import CliRunner
    runner = CliRunner()
    # Column does not exist
    result = runner.invoke(cugrep.cugrep, ["foo", str(file_path), "--column", "not_a_col"])
    assert result.exit_code == 0
    assert "No string columns to search" in result.output or result.output == ""

def test_cugrep_unsupported_filetype(tmp_path):
    file_path = tmp_path / "test.unsupported"
    file_path.write_text("irrelevant content")
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cugrep.cugrep, ["foo", str(file_path)])
    assert result.exit_code == 0
    assert "Unsupported file type" in result.output

def test_cugrep_regex_match(tmp_path):
    csv_content = "col1\nfoo123\nbar456\nfoo789\n"
    file_path = tmp_path / "test5.csv"
    file_path.write_text(csv_content)
    from click.testing import CliRunner
    runner = CliRunner()
    # Regex: match foo followed by digits
    result = runner.invoke(cugrep.cugrep, [r"foo\\d+", str(file_path), "--column", "col1"])
    assert result.exit_code == 0
    assert "foo123" in result.output
    assert "foo789" in result.output
    assert "bar456" not in result.output
