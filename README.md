# RiiSearcher

**RiiSearcher** is a vector similarity searcher based on the [`Rii`](https://github.com/matsui528/rii) library. Rii stands for `Reconfigurable Inverted Index` and is IVFPQ-based fast and memory efficient approximate nearest neighbor search method with a subset-search functionality.
It is a C++ library with Python bindings to search for points in space based on the well-known inverted file with product quantization (PQ) approach (so called IVFADC or IVFPQ)

For more information please refer to the Github [repo](https://github.com/matsui528/rii)

**NOTE:** Since we do not reconfigure Rii during `/index` request but during the `/search` request, therefore the first search after a CUD operation will always be slower. 

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
import os
import rii
import nanopq
import pickle
from jina import Flow, Document, DocumentArray

def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs

# First save the trained instance of RiiSearcher with the help
# of the helper function `train`

trained_index_file = os.path.join('.', 'rii.pkl')
train_data = np.array(np.random.random([1024, 10]), dtype=np.float32)
codec = nanopq.PQ(M=1, Ks=1, verbose=False).fit(vecs=train_data)
e = rii.Rii(fine_quantizer=codec) # create an instance of Rii using nanopq codec
with open(trained_index_file, 'wb') as f:
    pickle.dump(e, f)


# Now lets start using the trained Rii in the flow to directly index and search
index = np.array(np.random.random([1000, 10]), dtype=np.float32)
index_docs = _get_docs_from_vecs(index)

query = np.array(np.random.random([10, 10]), dtype=np.float32)
query_docs = _get_docs_from_vecs(query)


f = Flow().add(
    uses='jinahub+docker://RiiSearcher',
    uses_with={'model_path', 'path/to/rii.pkl'},
    timeout_ready=-1,
)

with f:
    f.post(on='/index', inputs=index_docs)
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
    f.post(on='/save', parameters={'model_path': '.'})

# Create a new flow to load the pre-trained indexer on start
f = Flow().add(
    uses='jinahub+docker://RiiSearcher',
    uses_with={'model_path', '.'},
    timeout_ready=-1
)

with f:
    # Now make a direct search
    f.post(on='/search', data=query_docs, parameters={'top_k': 4})
```