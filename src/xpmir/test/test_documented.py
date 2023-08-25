import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set
from docutils.nodes import document
from experimaestro.tools.documentation import undocumented
from docutils.parsers.rst import Parser, directives, Directive
from docutils.utils import new_document
from docutils.frontend import OptionParser
import docutils.nodes as nodes
from sphinx.directives.other import TocTree
from sphinx.domains.python import PyCurrentModule
from experimaestro.sphinx import PyObject


class autodoc(nodes.Node):
    def __init__(self, content) -> None:
        super().__init__()
        self.content = content
        self.children = []


class AutotocDirective(Directive):
    has_content = True
    optional_arguments = TocTree.optional_arguments
    option_spec = TocTree.option_spec
    required_arguments = TocTree.required_arguments

    def run(self):
        return [autodoc(self.content)]


class autoxpmconfig(nodes.Node):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.children = []


class AutoXPMDirective(Directive):
    has_content = PyObject.has_content
    optional_arguments = PyObject.optional_arguments
    required_arguments = PyObject.required_arguments
    option_spec = PyObject.option_spec.copy()
    option_spec.update({"members": directives.unchanged})

    def run(self):
        return [autoxpmconfig(self.arguments[0].strip())]


class currentmodule(nodes.Node):
    def __init__(self, modname: Optional[str]) -> None:
        super().__init__()
        self.modname = modname
        self.children = []


class CurrentModuleDirective(Directive):
    has_content = PyCurrentModule.has_content
    optional_arguments = PyCurrentModule.optional_arguments
    required_arguments = PyCurrentModule.required_arguments
    option_spec = PyCurrentModule.option_spec

    def run(self):
        modname = self.arguments[0].strip()
        if modname == "None":
            modname = None
        return [currentmodule(modname)]


@lru_cache
def get_parser():
    directives.register_directive("toctree", AutotocDirective)
    directives.register_directive("autoxpmconfig", AutoXPMDirective)
    directives.register_directive("currentmodule", CurrentModuleDirective)
    settings = OptionParser(components=(Parser,)).get_default_values()

    return Parser(), settings


class MyVisitor(nodes.NodeVisitor):
    def __init__(self, document: document):
        super().__init__(document)
        self.toc: Set[str] = set()
        self.currentmodule = None
        self.documented = set()

    def visit_autodoc(self, node: autodoc) -> None:
        self.toc.update(node.content)

    def visit_autoxpmconfig(self, node: autoxpmconfig) -> None:
        """Called for all other node types."""
        if node.name.find(".") == -1:
            name = f"{self.currentmodule}.{node.name}"
        else:
            name = node.name

        self.documented.add(name)
        logging.debug("[autoxpmconfig] %s / %s", node.name, name)

    def visit_currentmodule(self, node: currentmodule) -> None:
        """Called for all other node types."""
        self.currentmodule = node.modname
        logging.debug("[current module] %s", self.currentmodule)

    def unknown_visit(self, node: nodes.Node) -> None:
        """Called for all other node types."""
        pass


class DocumentVisitor:
    def __init__(self) -> None:
        self.processed = set()
        self.documented = set()
        self.errors = 0

    def parse_rst(self, doc_path: Path):
        logging.info("Parsing %s", doc_path)
        input_str = doc_path.read_text()
        parser, settings = get_parser()
        document = new_document(str(doc_path.absolute()), settings)
        parser.parse(input_str, document)
        visitor = MyVisitor(document)
        document.walk(visitor)
        self.documented.update(visitor.documented)

        for to_visit in visitor.toc:
            path = (doc_path.parent / f"{to_visit}.rst").absolute()
            if path not in self.processed:
                if path.is_file():
                    self.processed.add(path)
                    self.parse_rst(path)
                else:
                    logging.error(f"Could not find {to_visit} in file {doc_path}")
                    self.errors += 1


def test_documented():
    """Test if every configuration is documented"""
    doc_path = Path(__file__).parents[3] / "docs" / "source" / "index.rst"
    assert doc_path.is_file()

    visitor = DocumentVisitor()
    visitor.parse_rst(doc_path)
    assert visitor.errors == 0

    ok, configs = undocumented("xpmir", visitor.documented, set(["xpmir.test"]))
    assert ok, "There were some errors parsing xpmir modules"

    names = [f"{config.__module__}.{config.__qualname__}" for config in configs]
    for name in sorted(names):
        logging.error("%s is not documented", name)
    assert len(configs) == 0, "Undocumented configurations"
