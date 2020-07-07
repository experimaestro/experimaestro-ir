from pathlib import Path
from experimaestro import Choices
from datamaestro.definitions import data, argument
from datamaestro.data import Base


@argument("path", type=Path, help="Path to the index")
@argument("storePositions", default=False, help="Store term position within documents")
@argument("storeDocvectors", default=False, help="Store document term vectors")
@argument("storeRaw", default=False, help="Store raw document")
@argument(
    "storeContents",
    default=False,
    help="Store processed documents (e.g. with HTML tags)",
)
@argument(
    "stemmer",
    default="porter",
    checker=Choices(["porter", "krovetz", "none"]),
    help="The stemmer to use",
)
@data()
class Index(Base):
    pass
