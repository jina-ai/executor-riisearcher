__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path

import numpy as np
from jina import Document, DocumentArray, Executor

from executor import RiiSearcher

DOC_IDS_FILENAME = "doc_ids.bin"
RII_INDEX_FILENAME = "rii.pkl"
_DIM = 10

# fix the seed here
np.random.seed(500)
cur_dir = os.path.dirname(os.path.abspath(__file__))


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[0].parents[1] / 'config.yml'))
    assert ex.metric == 'euclidean'


def test_empty_train(rii_index):
    da = DocumentArray()
    rii_index.index(da)


def test_empty_index(rii_index):
    da = DocumentArray()
    rii_index.index(da)


def test_empty_search(rii_index):
    da = DocumentArray()
    rii_index.search(da)


def test_search_input_none(rii_index):
    rii_index.search(None)


def test_train(tmp_path, rii_index):
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    rii_index.train(
        vec,
        parameters={
            'dump_path': tmp_path,
        },
    )
    assert (tmp_path / RII_INDEX_FILENAME).is_file()


def test_train_and_index(tmp_path):
    # First train the indexer and save it
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    index = RiiSearcher()
    index.train(vec, parameters={'dump_path': tmp_path})

    # Load the pre-trained and index data
    da_index = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    rii_index = RiiSearcher(dump_path=str(tmp_path))
    rii_index.index(da_index, {})
    assert len(rii_index._doc_ids) == 1024
    assert rii_index._rii_index.N == 1024


def test_rii_search(trained_index, tmp_path):
    indexer = RiiSearcher(dump_path=str(tmp_path))
    vec = np.array(np.random.random([1024, 10]), dtype=np.float32)
    index_docs = _get_docs_from_vecs(vec)
    indexer.index(index_docs)

    vec = np.array(np.random.random([10, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(vec)
    indexer.search(query_docs)

    for q in query_docs:
        np.testing.assert_array_less(q.matches[0].scores['euclidean'].value, 10)


def test_save(trained_index, tmp_path):
    da = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    rii_index = RiiSearcher(dump_path=str(tmp_path))
    rii_index.index(da, {})
    rii_index.save(parameters={'dump_path': str(tmp_path)})

    assert (tmp_path / DOC_IDS_FILENAME).is_file()
    assert (tmp_path / RII_INDEX_FILENAME).is_file()
