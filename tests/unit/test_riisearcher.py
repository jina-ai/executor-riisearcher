__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path

import numpy as np
import pytest
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


def test_saved_rii_indexer(saved_rii, tmpdir):
    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(query)
    indexer = RiiSearcher(
        dump_path=tmpdir,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 4
    for d in query_docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            <= d.matches[1].scores[indexer.metric].value
        )


@pytest.mark.parametrize('max_num_points', [257, 500, None])
def test_train(tmpdir, max_num_points, rii_index):
    da = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    rii_index.train(
        da,
        parameters={
            'max_num_training_points': max_num_points,
        },
    )
    assert rii_index._is_trained
    assert rii_index._rii_index.N == 1024


def test_train_and_index(rii_index):
    NUM_DOCS = 1000
    da_train = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(NUM_DOCS)]
    )
    da_index = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(NUM_DOCS)]
    )
    rii_index.train(da_train)
    rii_index.index(da_index, {})
    assert len(rii_index._doc_ids) == 2 * NUM_DOCS
    assert rii_index._rii_index.N == 2 * NUM_DOCS


def test_rii_search(saved_rii, tmpdir):
    indexer = RiiSearcher(dump_path=tmpdir)
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(vec)
    indexer.search(query_docs)
    for q in query_docs:
        np.testing.assert_array_less(q.matches[0].scores['euclidean'].value, 10)


def test_save(tmp_path, rii_index):
    da = DocumentArray(
        [Document(embedding=np.random.random(_DIM)) for _ in range(1024)]
    )
    rii_index.train(da)
    rii_index.save(parameters={'dump_path': str(tmp_path)})

    assert (tmp_path / DOC_IDS_FILENAME).is_file()
    assert (tmp_path / RII_INDEX_FILENAME).is_file()
