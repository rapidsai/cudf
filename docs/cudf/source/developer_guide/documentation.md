# Writing documentation

cuDF documentation is split into multiple pieces.
All core functionality is documented using inline docstrings.
Additional pages like user or developer guides are written independently.
While docstrings are written using [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (reST),
the latter are written using [MyST](https://myst-parser.readthedocs.io/en/latest/)
The inline docstrings are organized using a small set of additional reST pages.
The results are all then compiled together using [Sphinx](https://www.sphinx-doc.org/en/master/).
This document discusses each of these components and how to contribute to them.

## Docstrings

cuDF docstrings use the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) style.
In lieu of a complete explanation,
we include here an example of the format and the commonly used sections:

```
class A:
    """Brief description of A.

    Longer description of A.

    Parameters
    ----------
    x : int
        Description of x, the first constructor parameter.
    """
    def __init__(self, x: int):
        pass

    def foo(self, bar: str):
        """Short description of foo.

        Longer description of foo.

        Parameters
        ----------
        bar : str
            Description of bar.

        Returns
        -------
        float
            Description of the return value of foo.

        Raises
        ------
        ValueError
            Explanation of when a ValueError is raised.
            In this case, a ValueError is raised if bar is "fail".

        Examples
        --------
        The examples section is _strongly_ encouraged.
        Where appropriate, it may mimic the examples for the corresponding pandas API.
        >>> a = A()
        >>> a.foo('baz')
        0.0
        >>> a.foo('fail')
        ...
        ValueError: Failed!
        """
        if bar == "fail":
            raise ValueError("Failed!")
        return 0.0
```

`numpydoc` supports a number of other sections of docstrings.
Developers should familiarize themselves with them, since many are useful in different scenarios.
Our guidelines include one addition to the standard the `numpydoc` guide.
Class properties, which are not explicitly covered, should be documented in the getter function.
That choice makes `help` more useful as well as enabling docstring inheritance in subclasses.

All of our docstrings are validated using [`ruff pydocstyle rules`](https://docs.astral.sh/ruff/rules/#pydocstyle-d).
This ensures that docstring style is consistent and conformant across the codebase.

## Published documentation

Documentation is compiled using Sphinx, which pulls docstrings from the code.
Rather than simply listing all APIs, however, we aim to mimic the pandas documentation.
To do so, we organize API docs into specific pages and sections.
These pages are stored in `docs/cudf/source/api_docs`.
For example, all `DataFrame` documentation is contained in `docs/cudf/source/api_docs/dataframe.rst`.
That page contains sections like "Computations / descriptive stats" to make APIs more easily discoverable.

Within each section, documentation is created using [`autosummary`](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
This plugin makes it easy to generate pages for each documented API.
To do so, each section of the docs looks like the following:

```
Section name
~~~~~~~~~~~~
.. autosummary::
   API1
   API2
   ...
```

Each listed will automatically have its docstring rendered into a separate page.
This layout comes from the [Sphinx theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) that we use.

````{note}
Under the hood, autosummary generates stub pages that look like this (using `cudf.concat` as an example):

```
cudf.concat
===========

.. currentmodule:: cudf

.. autofunction:: concat
```

Commands like `autofunction` come from [`autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
This directive will import cudf and pull the docstring from `cudf.concat`.
This approach allows us to do the minimal amount of manual work in organizing our docs,
while still matching the pandas layout as closely as possible.
````

When adding a new API, developers simply have to add the API to the appropriate page.
Adding the name of the function to the appropriate autosummary list is sufficient for it to be documented.

### Documenting classes

Python classes and the Sphinx plugins used in RAPIDS interact in nontrivial ways.
`autosummary`'s default page generated for a class uses [`autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to automatically detect and document all methods of a class.
That means that in addition to the manually created `autosummary` pages where class methods are grouped into sections of related features, there is another page for each class where all the methods of that class are automatically summarized in a table for quick access.
However, we also use the [`numpydoc`](https://numpydoc.readthedocs.io/) extension, which offers the same feature.
We use both in order to match the contents and style of the pandas documentation as closely as possible.

pandas is also particular about what information is included in a class's documentation.
While the documentation pages for the major user-facing classes like `DataFrame`, `Series`, and `Index` contain all APIs, less visible classes or subclasses (such as subclasses of `Index`) only include the methods that are specific to those subclasses.
For example, {py:class}`cudf.CategoricalIndex` only includes `codes` and `categories` on its page, not the entire set of `Index` functionality.

To accommodate these requirements, we take the following approach:
1. The default `autosummary` template for classes is overridden with a [simpler template that does not generate method or attribute documentation](https://github.com/rapidsai/cudf/blob/main/docs/cudf/source/_templates/autosummary/class.rst). In other words, we disable `autosummary`'s generation of Methods and Attributes lists.
2. We rely on `numpydoc` entirely for the classes that need their entire APIs listed (`DataFrame`/`Series`/etc). `numpydoc` will automatically populate Methods and Attributes section if (and only if) they are not already defined in the class's docstring.
3. For classes that should only include a subset of APIs, we include those explicitly in the class's documentation. When those lists exist, `numpydoc` will not override them. If either the Methods or Attributes section should be empty, that section must still be included but should simply contain "None". For example, the class documentation for `CategoricalIndex` could include something like the following:

```
    Attributes
    ----------
    codes
    categories

    Methods
    -------
    None

```

## Comparing to pandas

cuDF aims to provide a pandas-like experience.
However, for various reasons cuDF APIs may exhibit differences from pandas.
Where such differences exist, they should be documented.
We facilitate such documentation with the `pandas-compat` directive.
The directive should be used inside docstrings like so:

```
"""Brief

Docstring body

.. pandas-compat::
    **$API_NAME**

    Explanation of differences
```

All such API compatibility notes are collected and displayed in the rendered documentation.

## Writing documentation pages

In addition to docstrings, our docs also contain a number of more dedicated user guides.
These pages are stored in `docs/cudf/source/user_guide`.
These pages are all written using MyST, a superset of Markdown.
MyST allows developers to write using familiar Markdown syntax,
while also providing the full power of reST where needed.
These pages do not conform to any specific style or set of use cases.
However, if you develop any sufficiently complex new features,
consider whether users would benefit from a more complete demonstration of them.

```{note}
We encourage using links between pages.
We enable [Myst auto-generated anchors](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#auto-generated-header-anchors),
so links should make use of the appropriately namespaced anchors for links rather than adding manual links.

```

## Building documentation

### Requirements

The following are required to build the documentation:
- A RAPIDS-compatible GPU. This is necessary because the documentation execute code.
- A working copy of cudf in the same build environment.
  We recommend following the [build instructions](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#setting-up-your-build-environment).
- Sphinx, numpydoc, and MyST-NB.
  Assuming you follow the build instructions, these should automatically be installed into your environment.

### Building and viewing docs

Once you have a working copy of cudf, building the docs is straightforward:
1. Navigate to `/path/to/cudf/docs/cudf/`.
2. Execute `make html`

This will run Sphinx in your shell and generate outputs at `build/html/index.html`.
To view the results.
1. Navigate to `build/html`
2. Execute `python -m http.server`

Then, open a web browser and go to `https://localhost:8000`.
If something else is currently running on port 8000,
`python -m http.server` will automatically find the next available port.
Alternatively, you may specify a port with `python -m http.server $PORT`.

You may build docs on a remote machine but want to view them locally.
Assuming the other machine's IP address is visible on your local network,
you can view the docs by replacing `localhost` with the IP address of the host machine.
Alternatively, you may also forward the port using e.g.
`ssh -N -f -L localhost:$LOCAL_PORT:localhost:$REMOTE_PORT $REMOTE_IP`.
That will make `$REMOTE_IP:$REMOTE_PORT` visible at `localhost:$LOCAL_PORT`.

## Documenting cuDF internals

Unlike public APIs, the documentation of internal code (functions, classes, etc) is not linted.
Documenting internals is strongly encouraged, but not enforced in any particular way.
Regarding style, either full numpy-style docstrings or regular `#` comments are acceptable.
The former can be useful for complex or widely used functionality,
while the latter is fine for small one-off functions.
