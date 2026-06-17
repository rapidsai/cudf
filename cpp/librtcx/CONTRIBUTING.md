# Contributing to librtcx

Contributions to librtcx fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](https://github.com/rapidsai/cudf/issues/new/choose) describing the problem or new feature
   in detail.
2. To propose and implement a new feature, please file a new feature request
   [issue](https://github.com/rapidsai/cudf/issues/new/choose). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it, using the [code contributions](#code-contributions)
   guide below.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment.

As contributors and maintainers to this project, you are expected to abide by the
[Contributor Code of Conduct](https://docs.rapids.ai/resources/conduct/).

## Code contributions

1. Create a fork of the [cudf repository](https://github.com/rapidsai/cudf) and check out a branch
   with a name that describes your planned work.
2. Write code to address the issue or implement the feature.
3. Add unit tests.
4. [Create your pull request](https://github.com/rapidsai/cudf/compare).
5. Verify that CI passes all status checks. Fix if needed.
6. Wait for other developers to review your code and update code as needed.
7. Once reviewed and approved, a maintainer will merge your pull request.

If you are unsure about anything, don't hesitate to comment on issues and ask for clarification!

## Code Formatting

librtcx uses [pre-commit](https://pre-commit.com/) to execute code linters and formatters.
These tools ensure a consistent code format throughout the project.

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit:

```bash
pre-commit install
```
