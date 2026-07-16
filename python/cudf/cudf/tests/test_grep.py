# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cudfgrep GPU grep utility (``cudf.grep``)."""

from __future__ import annotations

import io

from cudf.grep import grep, main


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _texts(results):
    return [text for _, text in results]


def _linenos(results):
    return [lineno for lineno, _ in results]


# --------------------------------------------------------------------------- #
# Public grep() API: regex coverage
# --------------------------------------------------------------------------- #
def test_literal_match(tmp_path):
    f = _write(tmp_path, "log.txt", "apple pie\nbanana\napple tart\n")
    res = grep("apple", f)
    assert _texts(res) == ["apple pie", "apple tart"]
    assert _linenos(res) == [1, 3]


def test_anchor_start(tmp_path):
    f = _write(tmp_path, "log.txt", "ERROR boom\nan ERROR\nERROR again\n")
    res = grep("^ERROR", f)
    assert _texts(res) == ["ERROR boom", "ERROR again"]


def test_anchor_end(tmp_path):
    f = _write(tmp_path, "log.txt", "end here\nthe end\nend\n")
    res = grep("end$", f)
    assert _texts(res) == ["the end", "end"]


def test_digits(tmp_path):
    f = _write(tmp_path, "log.txt", "code 200\nno numbers\ncode 404\n")
    res = grep(r"\d+", f)
    assert _linenos(res) == [1, 3]


def test_char_class(tmp_path):
    f = _write(tmp_path, "log.txt", "a1\nbb\nc3\n")
    res = grep("[0-9]", f)
    assert _texts(res) == ["a1", "c3"]


def test_alternation(tmp_path):
    f = _write(tmp_path, "log.txt", "warn: x\ninfo: y\nerror: z\n")
    res = grep("warn|error", f)
    assert _linenos(res) == [1, 3]


def test_quantifier(tmp_path):
    f = _write(tmp_path, "log.txt", "ab\naaab\nb\n")
    res = grep("a+b", f)
    assert _texts(res) == ["ab", "aaab"]


def test_ignore_case(tmp_path):
    f = _write(tmp_path, "log.txt", "Error\nerror\nclean\n")
    res = grep("error", f, ignore_case=True)
    assert _linenos(res) == [1, 2]


def test_invert(tmp_path):
    f = _write(tmp_path, "log.txt", "keep\ndrop\nkeep\n")
    res = grep("drop", f, invert=True)
    assert _texts(res) == ["keep", "keep"]


def test_word_regexp(tmp_path):
    f = _write(tmp_path, "log.txt", "cat\ncategory\nthe cat sat\n")
    res = grep("cat", f, word=True)
    assert _texts(res) == ["cat", "the cat sat"]


def test_line_regexp(tmp_path):
    f = _write(tmp_path, "log.txt", "cat\ncat sat\na cat\n")
    res = grep("cat", f, whole_line=True)
    assert _texts(res) == ["cat"]


def test_only_matching_multiple_per_line(tmp_path):
    f = _write(tmp_path, "log.txt", "a1 b2 c3\nno digits\nx9\n")
    res = grep(r"\d", f, only_matching=True)
    assert res == [(1, "1"), (1, "2"), (1, "3"), (3, "9")]


def test_only_matching_with_invert_is_empty(tmp_path):
    # grep prints nothing for `-o -v`.
    f = _write(tmp_path, "log.txt", "a1\nbb\nc3\n")
    res = grep(r"\d", f, only_matching=True, invert=True)
    assert res == []


# --------------------------------------------------------------------------- #
# CLI main(): output formatting and exit codes
# --------------------------------------------------------------------------- #
def test_main_basic(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "alpha\nbeta\nalpha beta\n")
    rc = main(["alpha", f])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["alpha", "alpha beta"]


def test_main_line_numbers(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "one\ntwo\nthree\n")
    rc = main(["-n", "t", f])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["2:two", "3:three"]


def test_main_count(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "x\ny\nx\n")
    rc = main(["-c", "x", f])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "2"


def test_main_only_matching(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "a1 b2\nc3\n")
    rc = main(["-o", r"\d", f])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["1", "2", "3"]


def test_main_multiple_e_patterns(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "cat\ndog\nfish\n")
    rc = main(["-e", "cat", "-e", "dog", f])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["cat", "dog"]


def test_main_multiple_files_prefix(tmp_path, capsys):
    a = _write(tmp_path, "a.txt", "hit\nmiss\n")
    b = _write(tmp_path, "b.txt", "hit\n")
    rc = main(["hit", a, b])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == [f"{a}:hit", f"{b}:hit"]


def test_main_no_match_exit_code(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "a\nb\n")
    rc = main(["zzz", f])
    assert rc == 1
    assert capsys.readouterr().out == ""


def test_main_invalid_pattern_exit_code(tmp_path, capsys):
    f = _write(tmp_path, "log.txt", "a\n")
    rc = main(["(", f])
    assert rc == 2


def test_main_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("foo\nbar\nfoobar\n"))
    rc = main(["foo"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out == ["foo", "foobar"]
