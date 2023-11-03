# Benchmarks

We reproduce the duckdb (h2o) join and groupby benchmarks by doing the
following:

1. Pull latest from duckdblabs/db-benchmark
2. Build environments for pandas and polars: virtualenv pandas/py-pandas virtualenv polars/py-polars
3. Activate pandas virtualenv:

```bash
source pandas/py-pandas/bin/activate
```

4. Install cudf-private:

```bash
python -m pip install \
--extra-index-url=https://USERNAME:PASSWORD@pypi.k8s.rapids.ai/simple \
cudf-private-cu11
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

  1. Run Modified pandas benchmarks:

```bash
./_launcher/solution.R --solution=pandas --task=groupby --nrow=1e7
./_launcher/solution.R --solution=pandas --task=groupby --nrow=1e8
./_launcher/solution.R --solution=pandas --task=join --nrow=1e7
./_launcher/solution.R --solution=pandas --task=join --nrow=1e8
```
