"""
Pre-computed Anserini indices provided by Jimmy Lin (U. Waterloo)
"""

from datamaestro.definitions import argument, datatasks, datatags, dataset
from datamaestro.download.archive import tardownloader
from datamaestro.annotations.agreement import useragreement
from datamaestro.utils import HashCheck
from xpmir.index.anserini import Index
from hashlib import md5


@useragreement(
    """Robust04 dataset.
Please confirm you agree to the authors' data usage stipulations found at
https://trec.nist.gov/data/cd45/index.html"""
)
@tardownloader(
    "index",
    url="https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz",
    checker=HashCheck("15f3d001489c97849a010b0a4734d018", md5),
)
@dataset(Index)
def robust04(index):
    """Robust 2014 index

    Pre-computed Anserini index of the Robust 2014 collection; used parameters can be found
    at https://git.uwaterloo.ca/jimmylin/anserini-indexes/-/blob/master/index-robust04-20191213-readme.txt
    """
    return {
        "path": index,
        "storePositions": True,
        "storeDocvectors": True,
        "storeRaw": True,
        "storeContents": False,
        "stemmer": "porter",
    }
