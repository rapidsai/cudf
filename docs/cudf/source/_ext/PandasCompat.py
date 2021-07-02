# This file is adapted from official sphinx tutorial for `todo` extension:
# https://www.sphinx-doc.org/en/master/development/tutorials/PandasCompat.html

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


class PandasCompat(nodes.Admonition, nodes.Element):
    pass


class PandasCompatList(nodes.General, nodes.Element):
    pass


def visit_PandasCompat_node(self, node):
    self.visit_admonition(node)


def depart_PandasCompat_node(self, node):
    self.depart_admonition(node)


class PandasCompatListDirective(Directive):
    def run(self):
        return [PandasCompatList("")]


class PandasCompatDirective(SphinxDirective):

    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = "PandasCompat-%d" % self.env.new_serialno("PandasCompat")
        targetnode = nodes.target("", "", ids=[targetid])

        PandasCompat_node = PandasCompat("\n".join(self.content))
        PandasCompat_node += nodes.title(
            _("Pandas Compatibility Note"), _("Pandas Compatibility Note")
        )
        self.state.nested_parse(
            self.content, self.content_offset, PandasCompat_node
        )

        if not hasattr(self.env, "PandasCompat_all_pandas_compat"):
            self.env.PandasCompat_all_pandas_compat = []

        self.env.PandasCompat_all_pandas_compat.append(
            {
                "docname": self.env.docname,
                "lineno": self.lineno,
                "PandasCompat": PandasCompat_node.deepcopy(),
                "target": targetnode,
            }
        )

        return [targetnode, PandasCompat_node]


def purge_PandasCompats(app, env, docname):
    """On Sphinx hook: 
    """
    if not hasattr(env, "PandasCompat_all_pandas_compat"):
        return

    env.PandasCompat_all_pandas_compat = [
        PandasCompat
        for PandasCompat in env.PandasCompat_all_pandas_compat
        if PandasCompat["docname"] != docname
    ]


def merge_PandasCompats(app, env, docnames, other):
    if not hasattr(env, "PandasCompat_all_pandas_compat"):
        env.PandasCompat_all_pandas_compat = []
    if hasattr(other, "PandasCompat_all_pandas_compat"):
        env.PandasCompat_all_pandas_compat.extend(
            other.PandasCompat_all_pandas_compat
        )


def process_PandasCompat_nodes(app, doctree, fromdocname):
    if not app.config.include_pandas_compat:
        for node in doctree.traverse(PandasCompat):
            node.parent.remove(node)

    # Replace all PandasCompatList nodes with a list of the collected PandasCompats.
    # Augment each PandasCompat with a backlink to the original location.
    env = app.builder.env

    if not hasattr(env, "PandasCompat_all_pandas_compat"):
        env.PandasCompat_all_pandas_compat = []

    for node in doctree.traverse(PandasCompatList):
        if not app.config.include_pandas_compat:
            node.replace_self([])
            continue

        content = []

        for PandasCompat_info in env.PandasCompat_all_pandas_compat:
            para = nodes.paragraph()
            filename = env.doc2path(PandasCompat_info["docname"], base=None)
            description = _(
                "(The original entry is located in %s, line %d and can be found "
            ) % (filename, PandasCompat_info["lineno"])
            para += nodes.Text(description, description)

            # Create a reference
            newnode = nodes.reference("", "")
            innernode = nodes.emphasis(_("here"), _("here"))
            newnode["refdocname"] = PandasCompat_info["docname"]
            newnode["refuri"] = app.builder.get_relative_uri(
                fromdocname, PandasCompat_info["docname"]
            )
            newnode["refuri"] += "#" + PandasCompat_info["target"]["refid"]
            newnode.append(innernode)
            para += newnode
            para += nodes.Text(".)", ".)")

            # Insert into the PandasCompatList
            content.append(PandasCompat_info["PandasCompat"])
            content.append(para)

        node.replace_self(content)


def setup(app):
    app.add_config_value("include_pandas_compat", False, "html")

    app.add_node(PandasCompatList)
    app.add_node(
        PandasCompat,
        html=(visit_PandasCompat_node, depart_PandasCompat_node),
        latex=(visit_PandasCompat_node, depart_PandasCompat_node),
        text=(visit_PandasCompat_node, depart_PandasCompat_node),
    )

    app.add_directive("PandasCompat", PandasCompatDirective)
    app.add_directive("PandasCompatList", PandasCompatListDirective)
    app.connect("doctree-resolved", process_PandasCompat_nodes)
    app.connect("env-purge-doc", purge_PandasCompats)
    app.connect("env-merge-info", merge_PandasCompats)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
