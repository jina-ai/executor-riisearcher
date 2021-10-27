import os
import pickle

import nanopq
import numpy as np
import pytest
import rii

from executor import RiiSearcher

DOC_IDS_FILENAME = "doc_ids.bin"
RII_INDEX_FILENAME = "rii.pkl"


@pytest.fixture(scope='function')
def rii_index():
    return RiiSearcher()


@pytest.fixture(scope='function')
def trained_index(tmp_path):
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)
    index = RiiSearcher()
    index.train(vec, parameters={'dump_path': tmp_path})
    return index


@pytest.fixture(scope='function')
def saved_rii(tmpdir):
    trained_index_file = os.path.join(tmpdir, RII_INDEX_FILENAME)
    train_ids_file = os.path.join(tmpdir, DOC_IDS_FILENAME)
    train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
    ids = [idx for idx in range(1024)]
    codec = nanopq.PQ(M=1, Ks=1, verbose=False).fit(vecs=train_data)
    e = rii.Rii(fine_quantizer=codec)
    with open(trained_index_file, 'wb') as f:
        pickle.dump(e, f)
    with open(train_ids_file, 'wb') as f:
        pickle.dump(ids, f)
