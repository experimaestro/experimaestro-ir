from pathlib import Path
from experimaestro import Choices
from datamaestro.definitions import data, argument
from datamaestro.data import Base

@argument("path", type=Path)
@argument("storePositions", default=False)
@argument("storeDocvectors", default=False)
@argument("storeRaw", default=False)
@argument("storeContents", default=False)
@argument("stemmer", default="porter", checker=Choices(["porter", "krovetz", "none"]))
@data()
class Index(Base): 
    pass