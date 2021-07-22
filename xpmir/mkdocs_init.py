import re
from experimaestro.mkdocs.metaloader import DependencyInjectorFinder

DependencyInjectorFinder.install(
    re.compile(r"^(torch|faiss|pandas|bs4|pytorch_transformers|pytrec_eval|apex)($|\.)")
)
