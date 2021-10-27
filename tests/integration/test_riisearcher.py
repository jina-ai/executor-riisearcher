__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
from jina import Document, DocumentArray, Flow

from executor import RiiSearcher

_DIM = 10
DOC_IDS_FILENAME = "doc_ids.bin"
RII_INDEX_FILENAME = "rii.pkl"


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


def test_search_flow(tmp_path, saved_rii):
    index_docs = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(vec)

    f = Flow().add(
        uses=RiiSearcher,
        uses_with={'dump_path': str(tmp_path)},
        timeout_ready=-1,
    )
    with f:
        f.post(on='/index', inputs=index_docs)
        result = f.post(
            on='/search',
            inputs=query_docs,
            return_results=True,
            parameters={'top_k': 4},
        )[0].docs
        assert len(result[0].matches) == 4
        for d in result:
            assert (
                d.matches[0].scores['euclidean'].value
                <= d.matches[1].scores['euclidean'].value
            )


def test_save_load(tmp_path, saved_rii):
    da = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(vec)

    f = Flow().add(
        name='rii',
        uses=RiiSearcher,
        uses_with={'dump_path': str(tmp_path)},
        timeout_ready=-1,
    )
    with f:
        f.post(on='/save', target_peapod='rii', parameters={'dump_path': str(tmp_path)})
        assert (tmp_path / DOC_IDS_FILENAME).is_file()
        assert (tmp_path / RII_INDEX_FILENAME).is_file()
