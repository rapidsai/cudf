"""
cugrep: GPU-accelerated grep utility using cuDF
"""
import cudf
import click
import re

@click.command()
@click.argument('pattern')
@click.argument('filenames', nargs=-1)
@click.option('--column', '-c', default=None, help='Column to search (default: all string columns)')
@click.option('--filetype', '-t', default=None, help='File type: csv, parquet, json, orc, txt (default: auto)')
@click.option('--ignore-case', '-i', is_flag=True, help='Ignore case distinctions')
@click.option('--show-row', is_flag=True, help='Show row number in output')
def cugrep(pattern, filenames, column, filetype, ignore_case, show_row):
    """Search for PATTERN in each FILE using cuDF on the GPU."""
    flags = re.IGNORECASE if ignore_case else 0
    for fname in filenames:
        # Auto-detect file type if not specified
        ext = fname.split('.')[-1].lower()
        ftype = filetype or ext
        if ftype in ('csv', 'txt'):
            df = cudf.read_csv(fname)
        elif ftype == 'parquet':
            df = cudf.read_parquet(fname)
        elif ftype == 'json':
            df = cudf.read_json(fname)
        elif ftype == 'orc':
            df = cudf.read_orc(fname)
        else:
            click.echo(f"Unsupported file type: {ftype}")
            continue
        # Select columns to search
        if column:
            cols = [column] if column in df.columns else []
        else:
            cols = [col for col in df.columns if df[col].dtype == 'object']
        if not cols:
            click.echo(f"No string columns to search in {fname}")
            continue
        for col in cols:
            matches = df[col].str.contains(pattern, regex=True, case=not ignore_case, na=False)
            matched_rows = df[matches]
            for idx, row in matched_rows.iterrows():
                output = f"{fname}:"
                if show_row:
                    output += f"{idx}:"
                output += f"{row[col]}"
                click.echo(output)

if __name__ == '__main__':
    cugrep()
