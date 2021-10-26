# RiiSearcher

**RiiSearcher** is a vector similarity searcher based on the [`Rii`](https://github.com/matsui528/rii) library. Rii stands for `Reconfigurable Inverted Index` and is IVFPQ-based fast and memory efficient approximate nearest neighbor search method with a subset-search functionality.
It is a C++ library with Python bindings to search for points in space based on the well-known inverted file with product quantization (PQ) approach (so called IVFADC or IVFPQ)

For more information please refer to the Github [repo](https://github.com/matsui528/rii)

## Usage

Check [tests](tests) for an example on how to use it.

## Index and search

This example shows a common usage pattern where we first index some documents, and then
perform search on the index. 

Note that to achieve the desired trade-off between index and query
time on one hand, and search accuracy on the other, you will need to "finetune" the
index parameters. For more information on that, see [rii documentation](https://rii.readthedocs.io/en/latest/source/tips.html).


### Training

To use a trainable Rii indexer we can first train the indexer with the data from `train_data_file`:

```python
import numpy as np
from jina import Flow, Document, DocumentArray

def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs

train_data_file = 'train.npy'
train_data = np.array(np.random.random([10240, 256]), dtype=np.float32)
np.save(train_data_file, train_data)

index = np.array(np.random.random([1000, 10]), dtype=np.float32)
index_docs = _get_docs_from_vecs(index)

f = Flow().add(
    uses='jinahub://RiiSearcher',
    timeout_ready=-1,
)

with f:
    f.post(on='/train', parameters={'train_data_file': train_data_file})
    f.post(on='/index', inputs=index_docs)
    # the trained index will be dumped to "rii.pkl" at the `index_path`
    f.post(on='/dump', parameters={'index_path': '.'})
```

Then, we can directly use the trained indexer with providing `index_path` and search:

```python
query = np.array(np.random.random([10, 10]), dtype=np.float32)
query_docs = _get_docs_from_vecs(query)
f = Flow().add(
    uses='jinahub://RiiSearcher',
    timeout_ready=-1,
    uses_with={
      'index_path': '/path/to/index_file'
    },
)

with f:
    result = f.post(
        on='/search', data=query_docs, return_results=True, parameters={'top_k': 4}
    )
```