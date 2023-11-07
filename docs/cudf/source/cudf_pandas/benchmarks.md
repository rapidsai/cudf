# Benchmarks

## Database-like ops benchmarks

We reproduced the [Database-like ops benchmark](https://duckdblabs.github.io/db-benchmark/)
including a solution using `cudf.pandas`. Here are the results:

<figure>
<img src="../_static/duckdb-benchmark-groupby-join.png"
class="align-center" width="750"
alt="_static/duckdb-benchmark-groupby-join.png" />
<figcaption style="text-align: center;">Results of the <a
href="https://duckdblabs.github.io/db-benchmark/">Database-like ops
benchmark</a> including <span
class="title-ref">cudf.pandas</span>.</figcaption>
</figure>

**Note:** A missing bar in the results for a particular solution
indicates we ran into an error when executing one or more queries for
that solution.

You can see the per-query results [here](https://data.rapids.ai/duckdb-benchmark/index.html).

### Steps to reproduce

Below are the steps to reproduce the `cudf.pandas` results.  The steps
to reproduce the results for other solutions are documented in
<https://github.com/duckdblabs/db-benchmark#reproduce>.

1. Clone the latest <https://github.com/duckdblabs/db-benchmark>
2. Build environments for pandas:

```bash
virtualenv pandas/py-pandas
```

3. Activate pandas virtualenv:

```bash
source pandas/py-pandas/bin/activate
```

4. Install cudf:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12  # or cudf-cu11
```

5. Modify pandas join/group code to use `cudf.pandas`:

```bash
diff --git a/pandas/groupby-pandas.py b/pandas/groupby-pandas.py
index 58eeb26..2ddb209 100755
--- a/pandas/groupby-pandas.py
+++ b/pandas/groupby-pandas.py
@@ -1,4 +1,4 @@
-#!/usr/bin/env python3
+#!/usr/bin/env -S python3 -m cudf.pandas

 print("# groupby-pandas.py", flush=True)

 diff --git a/pandas/join-pandas.py b/pandas/join-pandas.py
 index f39beb0..655dd82 100755
 --- a/pandas/join-pandas.py
 +++ b/pandas/join-pandas.py
 @@ -1,4 +1,4 @@
 -#!/usr/bin/env python3
 +#!/usr/bin/env -S python3 -m cudf.pandas

  print("# join-pandas.py", flush=True)
```

6. Run Modified pandas benchmarks:

```bash
./_launcher/solution.R --solution=pandas --task=groupby --nrow=1e7
./_launcher/solution.R --solution=pandas --task=groupby --nrow=1e8
./_launcher/solution.R --solution=pandas --task=join --nrow=1e7
./_launcher/solution.R --solution=pandas --task=join --nrow=1e8
```
