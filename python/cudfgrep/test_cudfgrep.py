import os
import tempfile
import subprocess
import sys

import pytest

CUDFGREP = os.path.join(os.path.dirname(__file__), "cudfgrep.py")

@pytest.fixture
def sample_text_file():
    lines = [
        "abc 123 def",
        "456 ghi",
        "no numbers here",
        "abcABC",
        "",
        "special: !@#$$%"
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        yield f.name
    os.unlink(f.name)

def run_cudfgrep(args, env=None):
    cmd = [sys.executable, CUDFGREP] + args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def test_simple_regex(sample_text_file):
    out, err, code = run_cudfgrep(["-e", "abc", sample_text_file])
    assert "abc" in out
    assert code == 0

def test_ignore_case(sample_text_file):
    out, _, _ = run_cudfgrep(["-e", "abc", "-i", sample_text_file])
    assert "abcABC" in out

def test_count(sample_text_file):
    out, _, _ = run_cudfgrep(["-e", "\\d+", "-c", sample_text_file])
    assert out.isdigit()
    assert int(out) == 2

def test_multiple_matches(sample_text_file):
    out, _, _ = run_cudfgrep(["-e", "[a-z]+", sample_text_file])
    # Should match 'abc', 'def', 'ghi', 'no', 'numbers', 'here', 'abc', 'special'
    assert "abc" in out and "def" in out and "special" in out

def test_no_match(sample_text_file):
    out, _, _ = run_cudfgrep(["-e", "notfound", sample_text_file])
    assert out == ""

def test_gds_env(sample_text_file):
    env = os.environ.copy()
    env["CUDF_GDS"] = "1"
    out, _, _ = run_cudfgrep(["-e", "abc", sample_text_file], env=env)
    assert "abc" in out

def test_gds_flag(sample_text_file):
    out, _, _ = run_cudfgrep(["-e", "abc", "--gds", sample_text_file])
    assert "abc" in out
