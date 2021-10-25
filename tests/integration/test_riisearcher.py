__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from jina.executors.metas import get_default_metas
from jina_commons.indexers.dump import export_dump_streaming

from executor import RiiSearcher


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    metas['name'] = 'rii_idx'
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_train_and_index(metas, tmpdir):
    vec_idx = np.random.randint(0, high=512, size=[512]).astype(str)
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)

    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(query)

    train_data_file = os.path.join(os.environ['TEST_WORKSPACE'], 'train.npy')
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    np.save(train_data_file, train_data)

    export_dump_streaming(
        os.path.join(tmpdir, 'dump'),
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    dump_path = os.path.join(tmpdir, 'dump')

    f = Flow().add(
        uses=RiiSearcher,
        timeout_ready=-1,
    )
    with f:
        f.post(on='/train', parameters={'train_data_file': train_data_file})
        f.post(
            on='/dump',
            target_peapod='rii',
            parameters={
                'target_path': str(dump_path),
            },
        )

    f = Flow().add(
        uses=RiiSearcher,
        timeout_ready=-1,
        uses_with={
            'dump_path': dump_path,
        },
    )
    with f:
        result = f.post(
            on='/search', data=query_docs, return_results=True, parameters={'top_k': 4}
        )[0].docs
        assert len(result[0].matches) == 4
        for d in result:
            assert (
                d.matches[0].scores['euclidean'].value
                <= d.matches[1].scores['euclidean'].value
            )
