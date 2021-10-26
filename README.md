# RiiSearcher

**RiiSearcher** is a vector similarity searcher based on the [`Rii`](https://github.com/matsui528/rii) library. Rii stands for `Reconfigurable Inverted Index` and is IVFPQ-based fast and memory efficient approximate nearest neighbor search method with a subset-search functionality.
It is a C++ library with Python bindings to search for points in space based on the well-known inverted file with product quantization (PQ) approach (so called IVFADC or IVFPQ)

For more information please refer to the Github [repo](https://github.com/matsui528/rii)

## Usage

Check [tests](tests) for an example on how to use it.

## Index and search

This example shows a common usage pattern where we first train and index some documents, and then
perform search on the indexed data.

Note that to achieve the desired trade-off between index and query
time on one hand, and search accuracy on the other, you will need to "finetune" the
index parameters. For more information on that, see [rii documentation](https://rii.readthedocs.io/en/latest/source/tips.html).

To use a trainable Rii indexer we can first train the indexer and then make a search

```python
import numpy as np
from jina import Flow, Document, DocumentArray

def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs

index = np.array(np.random.random([1000, 10]), dtype=np.float32)
index_docs = _get_docs_from_vecs(index)

query = np.array(np.random.random([10, 10]), dtype=np.float32)
query_docs = _get_docs_from_vecs(query)

f = Flow().add(
    uses='jinahub+docker://RiiSearcher',
    timeout_ready=-1,
)

with f:
    f.post(on='/train', inputs=index_docs)
    
    # Now search for some data
    f.post(on='/search', data=query_docs, parameters={'top_k': 4})
```


## Save and load

We can also save the trained RiiSearcher with the indexed data. This example shows the how to save and then re-create the executor based on the saved index.

```python
f = Flow().add(
    uses='jinahub+docker://RiiSearcher',
    timeout_ready=-1,
)

with f:
    # Train and index data
    f.post(on='/train', inputs=index_docs)
    
    # Now save the trained indexer
    f.post(on='/save', parameters={'dump_path': '.'})

# Create a new flow to load the pre-trained indexer on start
f = Flow().add(
    uses='jinahub+docker://RiiSearcher',
    uses_with={'dump_path', '.'},
    timeout_ready=-1
)

with f:
    # Now make a direct search
    f.post(on='/search', data=query_docs, parameters={'top_k': 4})
```