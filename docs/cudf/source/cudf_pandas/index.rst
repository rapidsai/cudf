cudf.pandas
-----------

cuDF pandas accelerator mode (``cudf.pandas``) is built on cuDF and
**accelerates pandas code** on the GPU.  It supports **100% of the
Pandas API**, using the GPU for supported operations, and
automatically **falling back to pandas** for other operations.

.. code-block:: python

   %load_ext cudf.pandas
   # pandas API is now GPU accelerated

   import pandas as pd

   df = pd.read_csv("filepath")  # uses the GPU!
   df.groupby("col").mean()  # uses the GPU!
   df.rolling(window=3).sum()  # uses the GPU!
   df.apply(set, axis=1)  # uses the CPU (fallback)


.. figure:: ../_static/colab.png
    :width: 200px
    :target: https://nvda.ws/3BnjYjN

    Try it on Google Colab!

+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| **Zero Code Change Acceleration**                                                           | **Third-Party Library Compatible**                                                                                  |
|                                                                                             |                                                                                                                     |
| Just ``%load_ext cudf.pandas`` in Jupyter, or pass ``-m cudf.pandas`` on the command line.  | ``cudf.pandas`` is compatible with most third-party libraries that use pandas.                                      |
+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| **Run the same code on CPU or GPU**                                                         | **100% of the Pandas API**                                                                                          |
|                                                                                             |                                                                                                                     |
| Nothing changes, not even your `import` statements, when going from CPU to GPU.             | Combines the full flexibility of Pandas with blazing fast performance of cuDF                                       |
+---------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+


.. _cudf-pandas-jupyterlab-toggle:

.. admonition:: Enable from the JupyterLab toolbar, no %load_ext needed
   :class: tip

   In JupyterLab, you can also enable ``cudf.pandas`` from the GPU accelerator
   toggle, **without loading the extension**: you do not need to add
   ``%load_ext cudf.pandas`` or make any other changes to the notebook. This
   toggle requires the `jupyterlab-nvdashboard`_ extension to be installed.
   Once it is installed, open the **GPU Accel** menu in the notebook toolbar,
   select **cuDF pandas**, and restart the kernel when prompted. The option is
   shown only when cuDF is installed in the notebook's environment. See the
   `GPU Accelerators documentation`_ for a demo and details.

.. _jupyterlab-nvdashboard: https://github.com/rapidsai/jupyterlab-nvdashboard#installation
.. _GPU Accelerators documentation: https://github.com/rapidsai/jupyterlab-nvdashboard#gpu-accelerators

``cudf.pandas`` is available as part of the ``cudf`` package. See the
`installation and deployment guide
<https://docs.nvidia.com/datascience/install/#install-rapids>`_
to get up-and-running with cuDF.

See :doc:`usage` for other ways to enable ``cudf.pandas``, including from the
command line and programmatically with ``cudf.pandas.install()``.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage
   benchmarks
   how-it-works
   faq
   developer_guide
