# Usage

## Jupyter Notebooks and IPython

Load the `cudf.pandas` extension at the beginning of your
notebook. After that, just `import pandas` and operations will use the
GPU:

```python
%load_ext cudf.pandas

import pandas as pd

URL = "https://github.com/plotly/datasets/raw/master/tips.csv"
df = pd.read_csv(URL)                 # uses the GPU
df["size"].value_counts()             # uses the GPU
df.groupby("size").total_bill.mean()  # uses the GPU
df.apply(list, axis=1)                # uses the CPU (fallback)
```

## Command-line usage

From the command line, run your Python scripts with `-m cudf.pandas`:

```bash
python -m cudf.pandas script.py
```

## Understanding performance - the `cudf.pandas` profiler

`cudf.pandas` will attempt to use the GPU whenever possible and fall
back to CPU for certain operations. Running your code with the
`cudf.pandas.profile` magic generates a report showing which
operations used the GPU and which used the CPU. This can help you
identify parts of your code that could be rewritten to be more
GPU-friendly:

```python
%load_ext cudf.pandas
import pandas as pd
```

```python
%%cudf.pandas.profile
df = pd.DataFrame({'a': [0, 1, 2], 'b': [3,4,3]})

df.min(axis=1)
out = df.groupby('a').filter(
    lambda group: len(group) > 1
)
```

![cudf-pandas-profile](../_static/cudf-pandas-profile.png)

When an operation falls back to using the CPU, it's typically because
that operation isn't implemented by cuDF. The profiler generates a
handy link to report the missing functionality to the cuDF team.

To profile a script being run from the command-line, pass the
`--profile` argument:

```bash
python -m cudf.pandas --profile script.py
```
