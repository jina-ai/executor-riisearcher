__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pickle
from pathlib import Path

import nanopq
import numpy as np
import pytest
import rii
from jina import Document, DocumentArray, Executor
from jina.executors.metas import get_default_metas
from jina_commons.indexers.dump import export_dump_streaming

from executor import RiiSearcher


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


# fix the seed here
np.random.seed(500)
retr_idx = None
vec_idx = np.random.randint(0, high=512, size=[512]).astype(str)
vec = np.array(np.random.random([512, 10]), dtype=np.float32)

query = np.array(np.random.random([10, 10]), dtype=np.float32)
query_docs = _get_docs_from_vecs(query)

cur_dir = os.path.dirname(os.path.abspath(__file__))

_DIM = 10


@pytest.fixture(scope='function')
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    metas['name'] = 'rii_idx'
    yield metas
    del os.environ['TEST_WORKSPACE']


@pytest.fixture()
def tmpdir_dump(tmpdir):
    from jina_commons.indexers.dump import export_dump_streaming

    export_dump_streaming(
        os.path.join(tmpdir, 'dump'),
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    return os.path.join(tmpdir, 'dump')


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[0].parents[1] / 'config.yml'))
    assert ex.metric == 'euclidean'


def test_rii_indexer(metas, tmpdir_dump):
    trained_index_file = os.path.join(os.environ['TEST_WORKSPACE'], 'rii.pkl')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    codec = nanopq.PQ(M=1, Ks=1, verbose=True).fit(vecs=train_data)
    e = rii.Rii(fine_quantizer=codec)
    with open(trained_index_file, 'wb') as f:
        pickle.dump(e, f)

    indexer = RiiSearcher(
        prefetch_size=256,
        trained_index_file=trained_index_file,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(query_docs, parameters={'top_k': 4})
    assert len(query_docs[0].matches) == 4
    for d in query_docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            <= d.matches[1].scores[indexer.metric].value
        )


@pytest.mark.parametrize('train_data', ['new', 'none'])
@pytest.mark.parametrize('max_num_points', [None, 257, 500, 10000])
def test_indexer_train(metas, train_data, max_num_points, tmpdir):
    np.random.seed(500)
    num_data = 500
    num_dim = 64
    num_query = 10
    query = np.array(np.random.random([num_query, num_dim]), dtype=np.float32)
    vec_idx = np.random.randint(0, high=num_data, size=[num_data]).astype(str)
    vec = np.random.random([num_data, num_dim])

    dump_path = os.path.join(tmpdir, 'dump')
    export_dump_streaming(
        dump_path,
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    indexer = RiiSearcher(
        prefetch_size=256,
        max_num_training_points=max_num_points,
        metas=metas,
        dump_path=dump_path,
        runtime_args={'pea_id': 0},
    )

    query_docs = _get_docs_from_vecs(query)
    top_k = 4
    indexer.search(query_docs, parameters={'top_k': top_k})
    idx = query_docs.traverse_flat(['m']).get_attributes('id')
    dist = query_docs.traverse_flat(['m']).get_attributes('scores')

    assert len(idx) == len(dist)
    assert len(idx) == num_query * top_k


@pytest.mark.parametrize('max_num_points', [257, 500, None])
def test_rii_indexer_train(metas, tmpdir, max_num_points):
    train_data_file = os.path.join(os.environ['TEST_WORKSPACE'], 'train.npy')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    np.save(train_data_file, train_data)

    indexer = RiiSearcher(
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.train(
        parameters={
            'train_data_file': train_data_file,
            'max_num_training_points': max_num_points,
        }
    )
    assert indexer._is_trained


def test_train_before_index(metas, tmpdir_dump):
    indexer = RiiSearcher(
        prefetch_size=256,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    docs = _get_docs_from_vecs(query)
    indexer.search(docs, parameters={'top_k': 5})
    assert len(docs[0].matches) == 5
    for d in docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            <= d.matches[1].scores[indexer.metric].value
        )


def test_train_and_index(metas):
    train_data_file = os.path.join(os.environ['TEST_WORKSPACE'], 'train.npy')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    np.save(train_data_file, train_data)
    NUM_DOCS = 1000
    indexer = RiiSearcher(
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.train(
        parameters={
            'train_data_file': train_data_file,
        }
    )
    embeddings = np.random.random(size=(NUM_DOCS, _DIM))
    da1 = DocumentArray([Document(embedding=emb) for emb in embeddings])
    da2 = DocumentArray([Document(embedding=emb) for emb in embeddings])

    indexer.index(da1, {})
    assert len(indexer._doc_ids) == NUM_DOCS
    assert indexer._rii_index.N == NUM_DOCS
    assert len(indexer._vecs) == NUM_DOCS

    indexer.index(da2, {})
    assert len(indexer._doc_ids) == 2 * NUM_DOCS
    assert indexer._rii_index.N == 2 * NUM_DOCS
    assert len(indexer._vecs) == 2 * NUM_DOCS


def test_rii_searcher_empty(metas):
    q = np.array(np.random.random([10, 10]), dtype=np.float32)
    q_docs = _get_docs_from_vecs(q)
    indexer = RiiSearcher(
        metas=metas,
        runtime_args={'pea_id': 0},
        prefetch_size=256,
    )
    indexer.search(q_docs, parameters={'top_k': 4})
    assert len(q_docs[0].matches) == 0


def test_search_input_none(metas, tmpdir_dump):
    indexer = RiiSearcher(
        prefetch_size=256,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    indexer.search(None)


def test_rii_search(metas, tmpdir_dump):
    indexer = RiiSearcher(
        prefetch_size=256,
        dump_path=tmpdir_dump,
        metas=metas,
        runtime_args={'pea_id': 0},
    )
    query_docs = _get_docs_from_vecs(vec)
    indexer.search(query_docs)
    for q in query_docs:
        np.testing.assert_array_less(q.matches[0].scores['euclidean'].value, 10)


def test_dump(metas, tmpdir):
    num_data = 2
    num_dims = 64

    vecs = np.random.random((num_data, num_dims))
    keys = np.arange(0, num_data).astype(str)

    dump_path = os.path.join(tmpdir, 'dump')
    export_dump_streaming(
        dump_path,
        1,
        len(keys),
        zip(keys, vecs, [b'' for _ in range(len(vecs))]),
    )

    indexer = RiiSearcher(
        prefetch_size=256,
        metas=metas,
        dump_path=dump_path,
        runtime_args={'pea_id': 0},
    )

    indexer.dump()

    new_indexer = RiiSearcher(
        prefetch_size=256,
        metas=metas,
        runtime_args={'pea_id': 0},
    )

    assert new_indexer.size == 2

    query = np.zeros((1, num_dims))
    query[0, 1] = 5
    docs = _get_docs_from_vecs(query.astype('float32'))
    new_indexer.search(docs, parameters={'top_k': 2})
    for d in docs:
        assert (
            d.matches[0].scores[indexer.metric].value
            <= d.matches[1].scores[indexer.metric].value
        )
