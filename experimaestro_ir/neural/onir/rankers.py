from . import openNIR, Factory
import importlib
from experimaestro import config, argument


@argument("qlen", default=20)
@argument("dlen", default=2000)
@argument("add_runscore", default=False)
@config()
class Ranker(Factory):
    PACKAGE_NAME = "onir.rankers"


# --- DRMM


@argument("nbins", default=29, help="number of bins in matching histogram")
@argument(
    "hidden", default=5, help="hidden layer dimension for feed forward matching network"
)
@argument(
    "histType", default="logcount", help="histogram type: 'count', 'norm' or 'logcount'"
)
@argument("combine", default="idf", help="term gate type: 'sum' or 'idf'")
@config(openNIR.model.drmm)
class DRMM(Ranker):
    CLASS_NAME = "Drmm"
