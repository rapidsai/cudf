Regex IR documentation
======================

Regex IR compiles regular expressions into ordered Thompson Automata IR, typed
Instruction IR, and NVVM IR for linking into CUDA kernels.
The host interpreter is test-only; production execution uses libNVVM and
nvJitLink.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   usage
   architecture
   semantics
   ir
   codegen-guide
   corpus-report
   versioning
   api
