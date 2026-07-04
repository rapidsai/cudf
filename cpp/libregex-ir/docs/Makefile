# SPDX-License-Identifier: Apache-2.0

SPHINXOPTS  = -n -W --keep-going
SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = build

.PHONY: help html doxygen clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doxygen:
	@mkdir -p "$(BUILDDIR)/doxygen"
	@cd .. && doxygen docs/Doxyfile

html: doxygen
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	@rm -rf "$(BUILDDIR)"
