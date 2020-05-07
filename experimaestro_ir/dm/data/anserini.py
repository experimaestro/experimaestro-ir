from pathlib import Path
from datamaestro.definitions import data, argument
from datamaestro.data import Base

@argument("path", type=Path)
@data()
class Index(Base): 
    pass