---
name: Pandas Accelerator Mode Profiler Request
about: Request GPU support for a function executed on the CPU in pandas accelerator mode.
title: "[FEA]"
labels: "? - Needs Triage, feature request"
assignees: ''

---

This issue template is intended only to be used for requests stemming from usage of the profiler in pandas accelerator mode. If you'd like to file a general cuDF feature request, please [click here](https://github.com/rapidsai/cudf/issues/new?assignees=&labels=%3F+-+Needs+Triage%2C+feature+request&projects=&template=feature_request.md&title=%5BFEA%5D).

- id: which-functions
  type: textarea
  attributes:
    label: "Please copy/paste the list of functions that required CPU fallback from the profile summary report."
  validations:
    required: true

**Profiler Output**
If possible, please provide the full output of your profiling report.


**Additional context**
Add any other context, code examples, or references to existing implementations about the feature request here.
