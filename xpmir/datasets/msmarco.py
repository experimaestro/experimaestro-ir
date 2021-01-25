from tqdm import tqdm
from pytools import memoize_method
from onir.datasets.utils import TrecAssessedTopics
from onir import util, datasets, indices, log
from onir.interfaces import trec, plaintext
from onir.indices.sqlite import DocStore
from experimaestro import param, pathoption, config, task, configmethod
from datamaestro_text.data.ir.csv import AdhocDocuments as TSVAdhocDocuments
from datamaestro import prepare_dataset
from xpmir.anserini import Index as AnseriniIndex
from .index_backed import _init_indices_parallel, Dataset


# _SOURCES = {
#     # 'qidpidtriples.train.full': 'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tar.gz',
#     # seems the qidpidtriples.train.full link is broken... I'll host a mirror until they fix
#     'qidpidtriples.train.full': 'https://macavaney.us/misc/qidpidtriples.train.full.tar.gz',
#     'doctttttquery-predictions': 'https://storage.googleapis.com/doctttttquery_git/predicted_queries_topk_sampling.zip',
# }

# TODO: use custom subset
MINI_DEV = {
    "484694",
    "836399",
    "683975",
    "428803",
    "1035062",
    "723895",
    "267447",
    "325379",
    "582244",
    "148817",
    "44209",
    "1180950",
    "424238",
    "683835",
    "701002",
    "1076878",
    "289809",
    "161771",
    "807419",
    "530982",
    "600298",
    "33974",
    "673484",
    "1039805",
    "610697",
    "465983",
    "171424",
    "1143723",
    "811440",
    "230149",
    "23861",
    "96621",
    "266814",
    "48946",
    "906755",
    "1142254",
    "813639",
    "302427",
    "1183962",
    "889417",
    "252956",
    "245327",
    "822507",
    "627304",
    "835624",
    "1147010",
    "818560",
    "1054229",
    "598875",
    "725206",
    "811871",
    "454136",
    "47069",
    "390042",
    "982640",
    "1174500",
    "816213",
    "1011280",
    "368335",
    "674542",
    "839790",
    "270629",
    "777692",
    "906062",
    "543764",
    "829102",
    "417947",
    "318166",
    "84031",
    "45682",
    "1160562",
    "626816",
    "181315",
    "451331",
    "337653",
    "156190",
    "365221",
    "117722",
    "908661",
    "611484",
    "144656",
    "728947",
    "350999",
    "812153",
    "149680",
    "648435",
    "274580",
    "867810",
    "101999",
    "890661",
    "17316",
    "763438",
    "685333",
    "210018",
    "600923",
    "1143316",
    "445800",
    "951737",
    "1155651",
    "304696",
    "958626",
    "1043094",
    "798480",
    "548097",
    "828870",
    "241538",
    "337392",
    "594253",
    "1047678",
    "237264",
    "538851",
    "126690",
    "979598",
    "707766",
    "1160366",
    "123055",
    "499590",
    "866943",
    "18892",
    "93927",
    "456604",
    "560884",
    "370753",
    "424562",
    "912736",
    "155244",
    "797512",
    "584995",
    "540814",
    "200926",
    "286184",
    "905213",
    "380420",
    "81305",
    "749773",
    "850038",
    "942745",
    "68689",
    "823104",
    "723061",
    "107110",
    "951412",
    "1157093",
    "218549",
    "929871",
    "728549",
    "30937",
    "910837",
    "622378",
    "1150980",
    "806991",
    "247142",
    "55840",
    "37575",
    "99395",
    "231236",
    "409162",
    "629357",
    "1158250",
    "686443",
    "1017755",
    "1024864",
    "1185054",
    "1170117",
    "267344",
    "971695",
    "503706",
    "981588",
    "709783",
    "147180",
    "309550",
    "315643",
    "836817",
    "14509",
    "56157",
    "490796",
    "743569",
    "695967",
    "1169364",
    "113187",
    "293255",
    "859268",
    "782494",
    "381815",
    "865665",
    "791137",
    "105299",
    "737381",
    "479590",
    "1162915",
    "655989",
    "292309",
    "948017",
    "1183237",
    "542489",
    "933450",
    "782052",
    "45084",
    "377501",
    "708154",
}


def _iter_collection(path):
    logger = log.easy()
    with path.open("rt") as collection_stream:
        for did, text in logger.pbar(
            plaintext.read_tsv(collection_stream), desc="documents"
        ):
            yield indices.RawDoc(did, text)


@param("collection", type=TSVAdhocDocuments, help="tab-separated collection")
@pathoption("path", "docstore")
@task()
class BuildDocStore(DocStore):
    """Build doc-store from index"""

    def execute(self):
        idxs = [indices.SqliteDocstore(self.path)]
        _init_indices_parallel(idxs, _iter_collection(self.collection.path), True)


@param("stemmer", type=str, help="The stemmer to use")
@param("collection", type=TSVAdhocDocuments, help="tab-separated collection")
@pathoption("path", "index")
@task()
class Reindex(AnseriniIndex):
    """Build index (not stemmed) from index"""

    def execute(self):
        idxs = [indices.AnseriniIndex(self.path, stemmer=self.stemmer)]
        _init_indices_parallel(idxs, _iter_collection(self.collection.path), True)


@param("index_stem", type=AnseriniIndex)
@param("index", type=AnseriniIndex)
@param("docstore", type=DocStore)
@config()
class MsmarcoDataset(Dataset):
    """
    Interface to the MS-MARCO ranking dataset.
     > Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, RanganMajumder, and Li
     > Deng. 2016.  MS MARCO: A Human Generated MAchineReading COmprehension Dataset. InCoCo@NIPS.
    """

    SUBSETS = [
        "train",
        "train10",
        "train_med",
        "dev",
        "minidev",
        "judgeddev",
        "eval",
        "trec2019",
        "judgedtrec2019",
    ]

    def __init__(self):
        super().__init__()
        self.logger = log.easy()
        self.index = indices.AnseriniIndex(
            self.index.path, stemmer="none", name="fullindex"
        )
        self.index_stem = indices.AnseriniIndex(self.index_stem.path, name="stemindex")
        self.doc_store = indices.SqliteDocstore(self.docstore.path)

    def _get_docstore(self):
        return self.doc_store

    def _get_index(self, record):
        return self.index_stem

    def _get_index_for_batchsearch(self):
        return self.index_stem

    def qrels(self, fmt="dict"):
        return self._load_qrels(self.subset, fmt=fmt)

    @memoize_method
    def _load_qrels(self, fmt):
        return trec.read_qrels_fmt(str(self.assessed_topics.qrels_path()), fmt)

    def _load_queries_base(self):
        return self._load_topics()

    @memoize_method
    def _load_topics(self):
        return dict(
            self.logger.pbar(
                plaintext.read_tsv(self.assessed_topics.topics.path),
                desc="loading queries",
            )
        )

    @staticmethod
    def prepare():
        """Index the MS-Marco collection"""
        # Get the collection and index it
        collection = prepare_dataset("com.microsoft.msmarco.passage.collection")

        docstore = BuildDocStore(collection=collection).submit()
        index = Reindex(collection=collection, stemmer="none").submit()
        index_stem = Reindex(collection=collection, stemmer="porter").submit()

        def get(subset: str):
            topics = prepare_dataset(f"com.microsoft.msmarco.passage.{subset}.queries")
            qrels = prepare_dataset(f"com.microsoft.msmarco.passage.{subset}.qrels")
            assessed_topics = TrecAssessedTopics(topics=topics, assessments=qrels)
            return MsmarcoDataset(
                docstore=docstore,
                index=index,
                index_stem=index_stem,
                assessed_topics=assessed_topics,
            )

        return get
