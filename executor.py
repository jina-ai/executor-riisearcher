import os
import pickle
from typing import Dict, Tuple, Optional, List

import nanopq
import numpy as np
import rii
from jina import Executor, DocumentArray, requests, Document
from jina.logging.logger import JinaLogger

DOC_IDS_FILENAME = "doc_ids.bin"
RII_INDEX_FILENAME = "rii.pkl"


class RiiSearcher(Executor):
    """
    Rii-powered vector indexer and searcher
    or more information about the Rii
    supported parameters and installation problems, please consult:
        - https://github.com/matsui528/rii
    """

    def __init__(
        self,
        codewords: int = 1,
        candidates: int = 10,
        subspaces: int = 1,
        cluster_center: Optional[int] = None,
        iter: int = 5,
        default_top_k: int = 5,
        model_path: Optional[str] = None,
        traversal_paths: Tuple[str] = ('r',),
        is_verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param codewords: Number of codewords associated with each `D/M` subspace where D is the
                dimension of embedding array.
        :param candidates: The number of PQ-codes for the candidates of distance evaluation.
                With a higher ``L`` value, the accuracy is boosted but the runtime gets slower.
        :param subspaces: The number of subspaces for PQ/OPQ, which is basically the number of units
                into which the embeddings will be broken down to. It controls the runtime
                accuracy and memory consumptions
        :param cluster_center: The number of cluster centers. The default value is `None`, where
                `cluster_center` is set to `sqrt(N)` automatically with N being the number of index data
        :param iter: The number of iteration for pqk-means to update cluster centers
        :param model_path: the path containing the trained index file. Trained index file should
                be saved as `rii.pkl`.
        training points to training data from `train_filepath`.
            The points will be selected randomly from the available points
        :param traversal_paths: traverse path on docs, e.g. ['r'], ['c']
        :param default_top_k: get tok k vectors
        :param is_verbose: set to True for verbose output from Rii
        """
        super().__init__(*args, **kwargs)

        self.logger = JinaLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        self.cluster_center = cluster_center
        self.iter = iter
        self.metric = 'euclidean'
        self.subspaces = subspaces
        if codewords and codewords > 256:
            self.logger.warning(
                'Ks is greater than 256. Setting it to 256'
                ' so that each code must be uint8'
            )
            codewords = 256
        self.codewords = codewords
        self.candidates = candidates
        self.is_verbose = is_verbose
        self._doc_ids = []
        self.model_path = model_path
        self._is_trained = False
        self.default_top_k = default_top_k
        self.traversal_paths = traversal_paths

        self.codec = nanopq.PQ(
            M=self.subspaces, Ks=self.codewords, verbose=self.is_verbose
        )
        model_path = self.model_path or kwargs.get('runtime_args', {}).get(
            'model_path', None
        )
        if model_path:
            self.logger.info(f'Building RiiSearcher from model path: {model_path}')
            try:
                with open(os.path.join(model_path, RII_INDEX_FILENAME), "rb") as f:
                    self._rii_index = pickle.load(f)
                    self._rii_index.verbose = self.is_verbose
                    self._is_trained = True
            except FileNotFoundError:
                self.logger.warning(
                    'No snapshot of Rii indexer found, '
                    'you should train offline and build the indexer again!'
                )
        else:
            self.logger.warning(
                'No `model_path` provided, train offline and build the indexer from scratch!'
            )

    def _add_to_index(
        self, vectors: "np.ndarray", ids: List):
        if self._rii_index is None or not self._is_trained:
            self.logger.warning('Please train the indexer first before indexing data')
            return

        self._rii_index.add(vecs=vectors)
        self._needs_reconfigure = True
        self._doc_ids.extend(ids)

    def train(self, data: 'np.ndarray', parameters: {}, *args, **kwargs) -> None:
        """
        Helper method to only train nanopq codec to be used for Rii searcher. It
        saves the trained Rii Searcher to be re-used and does not contain any
        indexed data. The embedding data distribution should be same as the data
        embedding to be indexed
        Please do not use this with flow

        :param data: A numpy array with data to train
        :param parameters: Dictionary with optional parameters to override
        default parameters set at initialization. The only supported key is
            `model_path`, `code_words`, `sub_spaces`, and `model_path`.
        """
        if data is None or len(data) == 0:
            self.logger.warning('Please pass data for training')
            return

        codewords = parameters.get('codewords', self.codewords)
        subspaces = parameters.get('subspaces', self.subspaces)
        model_path = parameters.get('model_path', self.model_path)

        num_samples, _ = data.shape
        self.logger.info(f'Training nanopq codec with {num_samples} points')

        codec = nanopq.PQ(M=subspaces, Ks=codewords, verbose=self.is_verbose)
        codec.fit(vecs=data)
        rii_index = rii.Rii(fine_quantizer=codec)

        self.logger.info(f"Dumping the RiiSearcher to {model_path}")
        with open(os.path.join(model_path, RII_INDEX_FILENAME), 'wb') as f:
            pickle.dump(rii_index, f)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """Index the Documents' embeddings.
        :param docs: Documents whose `embedding` to index.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`, `cluster_center`, 'iter'.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        ids = flat_docs.get_attributes('id')
        embeddings = np.stack(flat_docs.embeddings).astype(np.float32)

        self._add_to_index(embeddings, ids)

    @requests(on="/search")
    def search(
        self,
        docs: DocumentArray,
        parameters: Optional[Dict] = {},
        *args,
        **kwargs,
    ):
        """Given the query document, run the approximate nearest neighbor search
        over the stored PQ-codes. This functions matches the identifiers and the
         distances of ``topk`` nearest PQ-codes to the query.
        Attach matches to the Documents in `docs`, each match containing only the
        `id` of the matched document and the `score`.

        :param docs: An array of `Documents` that should have the `embedding` property
            of the same dimension as vectors in the index
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. Supported keys are
            `traversal_paths`, `top_k`, `candidates`, and `target_ids`.
        """
        if not hasattr(self, '_rii_index'):
            self.logger.warning("Querying against an empty index")
            return

        if self._needs_reconfigure:
            cluster_center = parameters.get('cluster_center', self.cluster_center)
            iteration = parameters.get('iter', self.iter)
            self._rii_index.reconfigure(nlist=cluster_center, iter=iteration)

        if parameters is None:
            parameters = {}

        traversal_paths = parameters.get("traversal_paths", self.traversal_paths)
        target_ids = parameters.get("target_ids", None)
        top_k = int(parameters.get("top_k", self.default_top_k))
        candidates = parameters.get("candidates", self.candidates)

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._rii_index.query(
                q=doc.embedding,
                L=candidates,
                topk=top_k,
                target_ids=target_ids,
            )
            for idx, dist in zip(indices, dists):
                match = Document(id=self._doc_ids[idx])
                match.scores[self.metric] = dist
                doc.matches.append(match)

    @requests(on='/save')
    def save(self, parameters: Dict = {}, **kwargs):
        """
        Save a snapshot of the current indexer along with the indexed `Document` ids

        :param parameters: Dictionary with optional parameters to override
        default parameters set at initialization. The only supported key is
            `model_path`.
        """

        model_path = parameters.get('model_path', self.model_path)
        if model_path is None:
            raise ValueError(
                'The `model_path` must be provided to save the indexer state.'
            )

        os.makedirs(model_path, exist_ok=True)

        self.logger.info(f"Dumping the RiiSearcher to {model_path}")
        with open(os.path.join(model_path, RII_INDEX_FILENAME), 'wb') as f:
            pickle.dump(self._rii_index, f)

        with open(os.path.join(model_path, DOC_IDS_FILENAME), "wb") as fp:
            pickle.dump(self._doc_ids, fp)
