# cudf.pandas Guide

- A unified CPU/GPU experience that brings best-in-class performance to your
Pandas code
- Enabling people who love Pandas to go faster when Pandas gets slow with zero
code change

- Provides all of the Pandas API
  - Uses the GPU for supported operations
  - Uses the CPU otherwise


- Fallback to CPU if the GPU runs out of memory
- Currently compatible with most third-party libraries

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">CPU</th>
    <th class="tg-0pky">CPU+GPU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">

```python
import pandas as pd
df = pd.read_csv("filepath")
df.groupby(“col”).mean()
df.rolling(window=3).sum()
```

</td>
    <td class="tg-0pky">

```python
import pandas as pd
df = pd.read_csv("filepath")
df.groupby(“col”).mean()
df.rolling(window=3).sum()
```

</td>
  </tr>
</tbody>
</table>


> python -m cudf.autoload script.py or %load_ext cudf (for Notebooks)

## How is this different from other DataFrame-like libraries?

```{toctree}
:maxdepth: 1

best-practices
pandas-comparison
faq
developer-guide
example-notebook
```
