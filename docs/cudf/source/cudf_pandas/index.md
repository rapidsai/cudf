# cudf.pandas

<h2 style="color: #9234eb;">OPEN BETA: ACCELERATING PANDAS WITH ZERO CODE CHANGE</h2>

**Try it now**

<h3>cuDF Pandas Accelerator Mode</h3>
A unified CPU/GPU experience that brings best-in-class performance to your
Pandas workflows

<br/>
<style type="text/css">
.td {padding: 0 15px;}
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;color:#CED6DD;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;color:#CED6DD;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"><span style="font-weight:700;font-style:normal;text-decoration:none">Zero Code Change Acceleration</span><br><span style="font-weight:400;font-style:normal;text-decoration:none">Just load the cuDF IPython/Jupyter Notebook extension or use the cuDF Python module option.</span></th>
    <th class="tg-0lax"><span style="font-weight:700;font-style:normal;text-decoration:none">Third-Party Library Compatible</span><br><span style="font-weight:400;font-style:normal;text-decoration:none">Pandas accelerator mode is compatible with most third-party libraries that operate on Pandas objects.</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"><span style="font-weight:700;font-style:normal;text-decoration:none">One Codepath</span><br><span style="font-weight:400;font-style:normal;text-decoration:none">Develop, test, and run in production with a single codebase, regardless of hardware.</span></td>
    <td class="tg-0lax"><span style="font-weight:700;font-style:normal;text-decoration:none">Designed for When Pandas is Too Slow</span><br><span style="font-weight:400;font-style:normal;text-decoration:none">Keep using pandas rather than learning new frameworks or paradigms as your data grows. Just accelerate it on a GPU.</span></td>
  </tr>
</tbody>
</table>

<h3>Bringing the Speed of cuDF to Every Pandas User</h3>

``cudf.pandas`` can be usinged with IPython or Jupyter Notebooks or in standard Python scripts:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Python Script</th>
    <th class="tg-0pky">Notebook</th>
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

# python -m cudf.pandas script.py
```


</td>
    <td class="tg-0pky">

```python
%load_ext cudf.pandas

import pandas as pd
df = pd.read_csv("filepath")
df.groupby(“col”).mean()
df.rolling(window=3).sum()
```

</td>
  </tr>
</tbody>
</table>

<br/>

Loreum Ipsum Unlock the speed of NVIDIA GPUs and cuDF for Pandas workflows

![Fake](../_static/fake.png)



## How it works

- A unified CPU/GPU experience that brings best-in-class performance to your
Pandas code
- Enabling people who love Pandas to go faster when Pandas gets slow with zero
code change

- Provides all of the Pandas API
  - Uses the GPU for supported operations
  - Uses the CPU otherwise


- Fallback to CPU if the GPU runs out of memory
- Currently compatible with most third-party libraries
**Try it now**

## How is this different from other DataFrame-like libraries?

```{toctree}
:maxdepth: 1

pandas-comparison
best-practices
faq
developer-guide
example-notebook
```
