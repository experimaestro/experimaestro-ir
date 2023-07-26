from functools import lru_cache
from pathlib import Path
from experimaestro.tools.documentation import undocumented
from sphinx.parsers import RSTParser
from docutils.utils import new_document
from docutils.frontend import OptionParser
import docutils.nodes as nodes

def search_documented(node: nodes.Structural):
    for child in node.children:
        if isinstance(child, nodes.Structural):
            search_documented(child)
            print("** ", type(child), str(child)[20:])
        else:
            print("-- ", type(child), str(child)[20:])

@lru_cache
def get_parser():
    settings = OptionParser(
        components=(RSTParser,)
    ).get_default_values()

    return RSTParser(), settings


def parse_rst(doc_path: Path):
    input_str = doc_path.read_text()
    parser, settings = get_parser()
    document = new_document(str(doc_path.absolute()), settings)
    parser.parse(input_str, document)
    search_documented(document)


def test_documented():
    """Test if every configuration is documented"""
    doc_path = Path(__file__).parents[3] / "docs" / "source" / "index.rst"
    assert doc_path.is_file()
    parse_rst(doc_path)

    documented = []
    ok, configs = undocumented("xpmir", documented, set(["xpmir.test"]))