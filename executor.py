import os
import pickle
from typing import Dict, Tuple, Optional

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
        num_codeword: int = 1,
        num_candidates: int = 10,
        num_subspaces: int = 1,
        cluster_center: Optional[int] = None,
        iter_steps: int = 5,
        default_top_k: int = 5,
        max_num_training_points: Optional[int] = None,
        dump_path: Optional[str] = None,
        traversal_paths: Tuple[str] = ('r',),
        is_verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param codewords: Number of codewords associated with each `D/M` subspace where D is the
                dimension of embedding array. Typically 256.
        :param candidates: The number of PQ-codes for the candidates of distance evaluation.
                With a higher ``L`` value, the accuracy is boosted but the runtime gets slower.
        :param subspaces: The number of subspaces for PQ/OPQ, which is basically the number of units
                into which the embeddings will be broken down to. It controls the runtime
                accuracy and memory consumptions
        :param cluster_center: The number of cluster centers. The default value is `None`, where `nlist`
                is set to `sqrt(N)` automatically with N being the number of index data
        :param iter_steps: The number of iteration for pqk-means to update cluster centers
        :param dump_path: the path to load the trained index file and ids
        :param max_num_training_points: Optional argument to consider only a subset of
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
        self.max_num_training_points = max_num_training_points
        self.cluster_center = cluster_center
        self.iter = iter_steps
        self.metric = 'euclidean'
        self.subspaces = num_subspaces
        if num_codeword and num_codeword > 256:
            self.logger.warning(
                "Ks must be less than 256 so that each code must be uint8"
            )
        self.codewords = num_codeword  # Ks
        self.candidates = num_candidates  # L
        self.is_verbose = is_verbose
        self._doc_ids = []
        self.dump_path = dump_path
        self._is_trained = False
        self.default_top_k = default_top_k
        self.traversal_paths = traversal_paths

        self.codec = nanopq.PQ(
            M=self.subspaces, Ks=self.codewords, verbose=self.is_verbose
        )
        dump_path = self.dump_path or kwargs.get('runtime_args', {}).get(
            'dump_path', None
        )
        if dump_path:
            self.logger.info(f'Building RiiSearcher from dump data {dump_path}')
            try:
                with open(os.path.join(dump_path, RII_INDEX_FILENAME), "rb") as f:
                    self._rii_index = pickle.load(f)
                    self._rii_index.verbose = self.is_verbose
                    self._is_trained = True
            except FileNotFoundError:
                self.logger.info(
                    'No snapshot of Rii indexer found, '
                    'you should train and build the indexer from scratch!!'
                )
        else:
            self.logger.info(
                'No `dump_path` provided, train and build the indexer from scratch!!.'
            )

    def _add_to_index(
        self, vectors: "np.ndarray", ids: list, cluster_center: int, iter: int
    ):
        if self._rii_index is None or not self._is_trained:
            self.logger.warning('Please train the indexer first before indexing data')
            return

        self.logger.info('Building the Rii indexer...')
        self._rii_index.add_configure(vecs=vectors, nlist=cluster_center, iter=iter)
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
            `dump_path`, `code_words`, `sub_spaces`, and `dump_path`.
        """
        if data is None or len(data) == 0:
            self.logger.warning('Please pass data for training')
            return

        codewords = parameters.get('code_words', self.codewords)
        subspaces = parameters.get('sub_spaces', self.subspaces)
        dump_path = parameters.get('dump_path', self.dump_path)

        num_samples, _ = data.shape
        self.logger.info(f'Training nanopq codec with {num_samples} points')

        codec = nanopq.PQ(M=subspaces, Ks=codewords, verbose=self.is_verbose)
        codec.fit(vecs=data)
        rii_index = rii.Rii(fine_quantizer=codec)

        self.logger.info(f"Dumping the RiiSearcher to {dump_path}")
        with open(os.path.join(dump_path, RII_INDEX_FILENAME), 'wb') as f:
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

        cluster_center = parameters.get('cluster_center', self.cluster_center)
        iter = parameters.get('iter', self.iter)
        ids = flat_docs.get_attributes('id')
        embeddings = np.stack(flat_docs.embeddings).astype(np.float32)

        self._add_to_index(embeddings, ids, cluster_center, iter)

    @requests(on="/search")
    def search(
        self,
        docs: DocumentArray,
        parameters: Optional[Dict] = None,
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
            `traversal_paths`, `top_k`, `L`, and `target_ids`.
        """
        if not hasattr(self, '_rii_index'):
            self.logger.warning("Querying against an empty index")
            return

        if parameters is None:
            parameters = {}

        traversal_paths = parameters.get("traversal_paths", self.traversal_paths)
        target_ids = parameters.get("target_ids", None)
        top_k = int(parameters.get("top_k", self.default_top_k))
        L = parameters.get("L", self.candidates)

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._rii_index.query(
                q=doc.embedding,
                L=L,
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
            `dump_path`.
        """

        dump_path = parameters.get('dump_path', self.dump_path)
        if dump_path is None:
            raise ValueError(
                'The `dump_path` must be provided to save the indexer state.'
            )

        os.makedirs(dump_path, exist_ok=True)

        self.logger.info(f"Dumping the RiiSearcher to {dump_path}")
        with open(os.path.join(dump_path, RII_INDEX_FILENAME), 'wb') as f:
            pickle.dump(self._rii_index, f)

        with open(os.path.join(dump_path, DOC_IDS_FILENAME), "wb") as fp:
            pickle.dump(self._doc_ids, fp)
