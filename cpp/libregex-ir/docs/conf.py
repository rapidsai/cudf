from pathlib import Path

project = "Regex IR"
copyright = "2026, Regex IR contributors"
author = "Regex IR contributors"
version = "0.1"
release = "0.1.0"

extensions = ["breathe", "myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["build", "_build", "README.md"]

breathe_projects = {"regex_ir": str(Path(__file__).parent / "build" / "doxygen" / "xml")}
breathe_default_project = "regex_ir"
breathe_default_members = ("members",)
breathe_show_include = False

nitpicky = True
nitpick_ignore = [
    ("cpp:identifier", "std::invalid_argument"),
    ("cpp:identifier", "std::string"),
    ("cpp:identifier", "std::string_view"),
]

html_theme = "nvidia_sphinx_theme"
html_title = "Regex IR documentation"
