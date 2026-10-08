import pytest
import numpy as np
from pathlib import Path
from experimaestro import ObjectStore
from experimaestro.xpmutils import DirectoryContext
from xpmir.index.sparse import (
    Sparse2SeismicConverter,
    SparseRetrieverIndex,
)
from xpmir.test.utils.utils import SampleDocumentStore
import impact_index


@pytest.fixture
def context(tmp_path: Path):
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    return DirectoryContext(tmp_path)


def test_sparse2seismic_converter_and_retrieval(context, tmp_path):
    assert hasattr(impact_index, "SeismicSearcher")
    objects = ObjectStore()
    documents = SampleDocumentStore.C(num_docs=10)

    # 1. Build a basic impact index on disk
    raw_dir = tmp_path / "raw_index"
    builder = impact_index.IndexBuilder(str(raw_dir))
    builder.add(
        0,
        np.array([1, 2], dtype=np.uint64),
        np.array([1.5, 2.0], dtype=np.float32),
    )
    builder.add(
        1,
        np.array([2, 3], dtype=np.uint64),
        np.array([1.0, 3.0], dtype=np.float32),
    )
    builder.build(False)

    base_index_cfg = SparseRetrieverIndex.C(
        index_path=raw_dir,
        documents=documents,
    )

    # 2. Convert to Seismic
    converter = Sparse2SeismicConverter.C(
        index=base_index_cfg,
        n_postings=1000,
        summary_energy=1.0,
    )
    converter_instance = converter.instance(context=context, objects=objects)
    converter_instance.execute()
    seismic_index_cfg = converter.task_outputs(lambda x: x)

    # 3. Test Seismic index retrieval
    seismic_instance = seismic_index_cfg.instance(context=context, objects=objects)
    seismic_instance.initialize()

    results = seismic_instance.retrieve({2: 1.0}, top_k=2)
    assert len(results) == 2
    assert results[0].document["id"] == "0"
    assert results[0].score == pytest.approx(2.0, rel=1e-3)
    assert results[1].document["id"] == "1"
    assert results[1].score == pytest.approx(1.0, rel=1e-3)
