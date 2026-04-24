# cudfgrep

GPU-accelerated grep utility using cuDF/nvtext.

## Features
- Grep plain text files (one string per line, string only)
- Multiple matches per line are supported
- CLI matches grep: `-e`, `-i`, `-c`, etc.
- GDS enablement via `--gds` or `CUDF_GDS=1` environment variable
- Ready for future: column mode for CSV/Parquet, etc.

## Usage
```sh
python cudfgrep.py -e PATTERN [--gds] [file1.txt file2.txt ...]
```

- `-e`, `--regexp` : Pattern to search for (required)
- `-i`, `--ignore-case` : Ignore case distinctions
- `-c`, `--count` : Only print a count of matching lines per file
- `--gds` : Enable GDS (GPUDirect Storage)

## Benchmarking
- Run with and without `--gds` (or `CUDF_GDS=1`)
- Test on PCIe and Grace systems for throughput
- Example:
  ```sh
  time python cudfgrep.py -e 'error' biglog.txt
  CUDF_GDS=1 time python cudfgrep.py -e 'error' --gds biglog.txt
  ```

## Output
- Prints all matches per line (tab-separated)
- For `-c`, prints only the count of matching lines

## Example
```sh
python cudfgrep.py -e '\d+' sample.txt
```

## Requirements
- RAPIDS cuDF, nvtext, cupy

---
This is an experimental utility. Contributions welcome!
