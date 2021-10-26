import os
import pickle
from typing import Dict, Iterable, List, Optional

import nanopq
import numpy as np
import rii
from jina import Executor, DocumentArray, requests, Document
from jina.helper import batch_iterator
from jina.logging.logger import JinaLogger
from jina_commons.indexers.dump import import_vectors

DOC_IDS_FILENAME = "doc_ids.bin"
DOC_VECS_FILENAME = "doc_vecs.bin"
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
        Ks: Optional[int] = 1,
        L: Optional[int] = None,
        M: Optional[int] = 1,
        nlist: Optional[int] = None,
        iter_steps: Optional[iter] = 5,
        max_num_training_points: Optional[int] = None,
        index_path: Optional[str] = None,
        dump_path: Optional[str] = None,
        prefetch_size: Optional[int] = 512,
        traversal_paths: List[str] = ['r'],
        default_top_k: int = 5,
        is_verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param Ks: Number of codewords associated with each `D/M` subspace where D is the
                dimension of embedding array. Typically 256.
        :param L: The number of PQ-codes for the candidates of distance evaluation.
                With a higher ``L`` value, the accuracy is boosted but the runtime gets slower.
        :param M: The number of subspaces for PQ/OPQ, which is basically the number of units
                into which the embeddings will be broken down to. It controls the runtime
                accuracy and memory consumptions
        :param nlist: The number of cluster centers. The default value is `None`, where `nlist`
                is set to `sqrt(N)` automatically with N being the number of index data
        :param iter_steps: The number of iteration for pqk-means to update cluster centers
        :param dump_path: the path to load ids and vecs
        :param index_path: the path to load the trained index file if no `dump_path`
        :param max_num_training_points: Optional argument to consider only a subset of
        training points to training data from `train_filepath`.
            The points will be selected randomly from the available points
        :param prefetch_size: the number of data to pre-load into RAM
        :param traversal_paths: traverse path on docs, e.g. ['r'], ['c']
        :param default_top_k: get tok k vectors
        :param is_verbose: set to True for verbose output from Rii
        """
        super().__init__(*args, **kwargs)

        self.logger = JinaLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        self.max_num_training_points = max_num_training_points
        self.prefetch_size = prefetch_size
        self.nlist = nlist
        self.iter = iter_steps
        self.metric = 'euclidean'
        self.M = M
        if Ks and Ks > 256:
            self.logger.warning(
                "Ks must be less than 256 so that each code must be uint8"
            )
        self.Ks = Ks
        self.L = L
        self.is_verbose = is_verbose

        self._doc_ids = []
        self._vecs = []
        self._doc_id_to_offset = {}

        self._is_deleted = []
        self._prefetch_data = []
        self._train_data = []

        self._is_trained = False
        self.default_top_k = default_top_k
        self.traversal_paths = traversal_paths
        self.codec = nanopq.PQ(M=self.M, Ks=self.Ks, verbose=True)
        if dump_path:
            self.logger.info('Starting to build RiiSearcher from dump data')
            self._load_dump(dump_path, **kwargs)
        else:
            self.logger.info(
                'No `dump_path` provided, attempting to load pre-trained indexer.'
            )
            self._load(index_path)

    def _load_dump(self, dump_path, **kwargs):
        if dump_path is not None:
            self.logger.info(f'Start building "RiiIndexer" from dump data {dump_path}')
            ids_iter, vecs_iter = import_vectors(
                dump_path, str(self.runtime_args.pea_id)
            )
            iterator = zip(ids_iter, vecs_iter)
        else:
            self.logger.warning('No data loaded in "RiiIndexer".')
            return

        if iterator is not None:
            iterator = self._iterate_vectors_and_save_ids(iterator)
            self._build_index(iterator)

    def _build_index(self, vecs_iter: Iterable["np.ndarray"]):
        """Build an advanced index structure from a numpy array.

        :param vecs_iter: iterator of numpy array containing the vectors to index
        """
        if self.max_num_training_points is None:
            self.logger.info("Taking complete indexed data as training points...")
            self._prefetch_data.extend(list(vecs_iter))
        else:
            self.logger.info("Taking partial indexed data as training points")
            while (
                self.max_num_training_points
                and len(self._prefetch_data) < self.max_num_training_points
            ):
                try:
                    self._prefetch_data.append(next(vecs_iter))
                except StopIteration as _:
                    break

        if len(self._prefetch_data) == 0:
            return

        train_data = np.stack(self._prefetch_data)
        train_data = train_data.astype(np.float32)

        self.logger.info("Training the codec for Rii indexer...")
        self._train(train_data)

        self.logger.info("Building the Rii index...")
        self._build_rii_index(vecs_iter)

    def _train(self, data: "np.ndarray", *args, **kwargs) -> None:
        _num_samples, _ = data.shape
        self.logger.info(f"Training Rii Indexer with {_num_samples} points")
        self.codec.fit(vecs=data)
        self._rii_index = rii.Rii(fine_quantizer=self.codec)
        self._rii_index.verbose = self.is_verbose
        self._is_trained = True

    def _build_rii_index(self, vecs_iter: Iterable['np.ndarray']):
        if len(self._prefetch_data) > 0:
            vecs = np.stack(self._prefetch_data).astype(np.float32)
            self._index(vecs)
            self._prefetch_data.clear()

        for batch_data in batch_iterator(vecs_iter, self.prefetch_size):
            batch_data = list(batch_data)
            if len(batch_data) == 0:
                break
            vecs = np.stack(batch_data).astype(np.float32)
            self._index(vecs)

    def _index(self, vecs: "np.ndarray"):
        self._rii_index.add_configure(vecs=vecs, nlist=self.nlist, iter=self.iter)

    def _load(self, from_path: Optional[str] = None):
        from_path = from_path if from_path else self.workspace
        self.logger.info(f'Trying to restore indexer from {from_path}...')
        try:
            with open(os.path.join(from_path, RII_INDEX_FILENAME), "rb") as f:
                self._rii_index = pickle.load(f)
                self._rii_index.verbose = self.is_verbose
                self._is_trained = True

            with open(os.path.join(from_path, DOC_IDS_FILENAME), 'rb') as fp:
                self._doc_ids = pickle.load(fp)

            with open(os.path.join(from_path, DOC_VECS_FILENAME), 'rb') as fp:
                self._vecs = pickle.load(fp)

        except FileNotFoundError:
            self.logger.warning(
                'No snapshot of Rii indexer found, '
                'you should train and build the indexer from scratch!!'
            )
            return False
        except Exception as ex:
            raise ex

        return True

    @requests(on="/train")
    def train(self, parameters: Dict, **kwargs):
        """
        Train the nanopq codec for Rii initialisation. Currently it accepts .npq file
        which contains the embedding vectors with the same distribution as the data to be indexed

        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `train_data_file`, and `max_num_training_points`.
        :param kwargs:
        :return:
        """
        train_data_file = parameters.get("train_data_file")
        if train_data_file is None:
            raise ValueError(f'No "train_data_file" provided for training {self}')

        max_num_training_points = parameters.get(
            'max_num_training_points', self.max_num_training_points
        )

        train_data = self._load_training_data(train_data_file)
        if train_data is None:
            raise ValueError(
                "Loading training data failed. Rii Searcher requires training data."
            )

        train_data = train_data.astype(np.float32)

        if max_num_training_points and max_num_training_points < train_data.shape[0]:
            self.logger.warning(
                f'From train_data with num_points {train_data.shape[0]}, '
                f'sample {max_num_training_points} points'
            )
            random_indices = np.random.choice(
                train_data.shape[0],
                size=min(max_num_training_points, train_data.shape[0]),
                replace=False,
            )
            train_data = train_data[random_indices, :]

        self.logger.info("Training the codec and initializing Rii indexer...")
        self._train(train_data)

    def _load_training_data(self, train_filepath: str) -> 'np.ndarray':
        self.logger.info(f'Loading training data from {train_filepath}')
        result = None

        try:
            result = np.load(train_filepath)
            if isinstance(result, np.lib.npyio.NpzFile):
                self.logger.warning(
                    '.npz format is not supported. Please save the array in .npy '
                    'format.'
                )
                result = None
        except Exception as e:
            self.logger.error(
                'Loading training data with np.load failed, filepath={}, {}'.format(
                    train_filepath, e
                )
            )

        if result is None:
            try:
                # Read from binary file:
                with open(train_filepath, 'rb') as f:
                    result = f.read()
            except Exception as e:
                self.logger.error(
                    'Loading training data from binary'
                    ' file failed, filepath={}, {}'.format(train_filepath, e)
                )
        return result

    @requests(on='/index')
    def index(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Index the Documents' embeddings.
        :param docs: Documents whose `embedding` to index.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """

        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        ids = flat_docs.get_attributes('id')

        self._append_vecs_and_ids(ids, flat_docs.embeddings)

    @requests(on="/search")
    def search(
        self,
        docs: Optional[DocumentArray],
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
            `traversal_paths`, `top_k` and `target_ids`.
        """
        if docs is None:
            return
        if not hasattr(self, "_rii_index"):
            self.logger.warning("Querying against an empty index")
            return

        if parameters is None:
            parameters = {}

        traversal_paths = parameters.get("traversal_paths", self.traversal_paths)
        target_ids = parameters.get("target_ids", None)
        top_k = int(parameters.get("top_k", self.default_top_k))
        L = parameters.get("L", self.L)

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._rii_index.query(
                q=doc.embedding,
                L=L,
                topk=top_k,
                target_ids=target_ids,
            )
            for idx, dist in zip(indices, dists):
                match = Document(
                    id=self._doc_ids[idx], embedding=np.array(self._vecs[idx])
                )
                match.scores[self.metric] = dist
                doc.matches.append(match)

    def _iterate_vectors_and_save_ids(self, iterator):
        for position, id_vector in enumerate(iterator):
            id_ = id_vector[0]
            vector = id_vector[1]
            self._doc_ids.append(id_)
            self._vecs.append(vector)
            self._doc_id_to_offset[id_] = position
            self._is_deleted.append(0)
            if vector is not None:
                # this should already be a np.array, NOT bytes
                yield vector
            else:
                yield None

    def _append_vecs_and_ids(self, doc_ids: List[str], vecs: np.ndarray):
        assert len(doc_ids) == vecs.shape[0]
        vecs = vecs.astype(np.float32)
        for doc_id, vec in zip(doc_ids, vecs):
            self._vecs.append(vec)
            self._doc_ids.append(doc_id)
            self._is_deleted.append(0)
        self._index(vecs)

    @requests(on='/dump')
    def dump(self, parameters: Dict = {}, **kwargs):
        """
        Save a snapshot of the current indexer
        """

        target_path = (
            parameters['index_path'] if 'index_path' in parameters else self.workspace
        )

        os.makedirs(target_path, exist_ok=True)

        self.logger.info(f"Dumping the RiiSearcher to {target_path}")
        with open(os.path.join(target_path, RII_INDEX_FILENAME), 'wb') as f:
            pickle.dump(self._rii_index, f)

        with open(os.path.join(target_path, DOC_IDS_FILENAME), "wb") as fp:
            pickle.dump(self._doc_ids, fp)

        with open(os.path.join(target_path, DOC_VECS_FILENAME), "wb") as fp:
            pickle.dump(self._doc_ids, fp)

    @property
    def size(self):
        """Return the nr of elements in the index"""
        return len(self._doc_ids) - self.deleted_count

    @property
    def deleted_count(self):
        return sum(self._is_deleted)
