import os
import pickle
import datetime
from typing import Callable, List, Optional

import nanopq
import numpy as np
from jina.helper import batch_iterator
import io
import rii
from jina.logging.logger import JinaLogger
from jina import Executor, DocumentArray, requests, Document

from jina_commons.indexers.dump import import_vectors

from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple


DOC_IDS_FILENAME = "doc_ids.bin"
RII_INDEX_FILENAME = "rii.pkl"


class RiiSearcher(Executor):
    def __init__(
        self,
        Ks: Optional[int] = None,
        L: Optional[int] = None,
        M: Optional[int] = 32,
        nlist: Optional[int] = None,
        iter_steps: Optional[iter] = 5,
        trained_index_file: Optional[str] = None,
        max_num_training_points: Optional[int] = None,
        dump_path: Optional[str] = None,
        dump_func: Optional[Callable] = None,
        prefetch_size: Optional[int] = 512,
        default_traversal_paths: List[str] = ['r'],
        default_top_k: int = 5,
        *args,
        **kwargs,
    ):
        """
        :param Ks: Number of codewords associated with each `D/M` subspace where D is the
                dimension of embedding array. Typically 256.
        :param L: The number of PQ-codes for the candidates of distance evaluation.
                With a higher ``L`` value, the accuracy is boosted but the runtime gets slower.
        :param M: The number of subspaces for PQ/OPQ, which is basically the number of units i
                nto which the embeddings will be broken down to. It controls the runtime
                accuracy and memory consumptions
        :param nlist: The number of cluster centers. The default value is `None`, where `nlist`
                is set to `sqrt(N)` automatically with N being the number of index data
        :param iter_steps: The number of iteration for pqk-means to update cluster centers
        :param trained_index_file: the index file dumped from a trained
            index, e.g., ``rii.pkl``. If none is provided, `indexed` data will be used
            to train the Indexer (In that case, one must be careful when sharding
            is enabled, because every shard will be trained with its own part of data).
        :param max_num_training_points: Optional argument to consider only a subset of
        training points to training data from `train_filepath`.
            The points will be selected randomly from the available points
        :param prefetch_size: the number of data to pre-load into RAM
        :param default_traversal_paths: traverse path on docs, e.g. ['r'], ['c']
        :param default_top_k: get tok k vectors
        """
        super().__init__(*args, **kwargs)

        self.logger = JinaLogger(
            getattr(self.metas, "name", self.__class__.__name__)
        ).logger
        self.trained_index_file = trained_index_file
        self.max_num_training_points = max_num_training_points
        self.prefetch_size = prefetch_size
        self.nlist = nlist
        self.iter = iter_steps
        self.metric = 'euclidean'
        self.M = M
        if Ks <= 256:
            self.logger.warning(
                "Ks must be less than 256 so that each code must be uint8"
            )
        self.Ks = Ks
        self.L = L

        self._doc_ids = []
        self._vecs = []
        self._doc_id_to_offset = {}

        self._is_deleted = []
        self._prefetch_data = []

        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.codec = nanopq.PQ(M=self.M, Ks=self.Ks, verbose=True)
        if dump_path or dump_func:
            self._load_dump(dump_path, dump_func, prefetch_size, **kwargs)
        else:
            self._load(self.workspace)

    def _load_dump(self, dump_path, dump_func, prefetch_size, **kwargs):
        if dump_path is not None:
            self.logger.info(f'Start building "RiiIndexer" from dump data {dump_path}')
            ids_iter, vecs_iter = import_vectors(
                dump_path, str(self.runtime_args.pea_id)
            )
            iterator = zip(ids_iter, vecs_iter)
        elif dump_func is not None:
            iterator = dump_func(shard_id=self.runtime_args.pea_id)
        else:
            self.logger.warning('No "dump_path" or "dump_func" passed to "RiiIndexer".')
            return

        if iterator is not None:
            iterator = self._iterate_vectors_and_save_ids(iterator)
            self._build_index(iterator)

    def _build_index(self, vecs_iter: Iterable["np.ndarray"]):
        """Build an advanced index structure from a numpy array.

        :param vecs_iter: iterator of numpy array containing the vectors to index
        """

        if self.trained_index_file and os.path.exists(self.trained_index_file):
            with open(self.trained_index_file, "rb") as f:
                self._rii_index = pickle.load(f)
        else:
            self.logger.info("Taking indexed data as training points...")
            if self.max_num_training_points is None:
                self._prefetch_data.extend(list(vecs_iter))
            else:
                self.logger.info("Taking indexed data as training points")
                while (
                    self.max_num_training_points
                    and len(self._prefetch_data) < self.max_num_training_points
                ):
                    try:
                        self._prefetch_data.append(next(vecs_iter))
                    except Exception as _:  # noqa: F841
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
        self.logger.info(f"Training faiss Indexer with {_num_samples} points")
        self.codec.fit(vecs=data)
        self._rii_index = rii.Rii(fine_quantizer=self.codec)

    def _build_rii_index(self, vecs_iter: Iterable["np.ndarray"]):
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
        except FileNotFoundError:
            self.logger.warning(
                'No snapshot of Rii indexer found, '
                'you should build the indexer from scratch!!'
            )
            return False
        except Exception as ex:
            raise ex

        return True

    @requests(on="/train")
    def train(self, parameters: Dict, **kwargs):
        train_data_file = parameters.get("train_data_file")
        if train_data_file is None:
            raise ValueError(f'No "train_data_file" provided for training {self}')

        max_num_training_points = parameters.get(
            'max_num_training_points', self.max_num_training_points
        )
        trained_index_file = parameters.get(
            "trained_index_file", self.trained_index_file
        )
        if not trained_index_file:
            raise ValueError(f'No "trained_index_file" provided for training {self}')

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

        self._train(train_data)

        self.logger.info(f"Dumping the trained Rii index to {trained_index_file}")
        if os.path.exists(trained_index_file):
            self.logger.warning(
                f"We are going to overwrite the index file located at "
                f"{trained_index_file}"
            )
        with open(trained_index_file, 'rb') as f:
            pickle.dump(self._rii_index, f)

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

    @requests(on="/search")
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Given a query vector, run the approximate nearest neighbor search over the stored PQ-codes.
        This functions returns the identifiers and the distances of ``topk`` nearest PQ-codes to the query.
        """
        if not hasattr(self, "_rii_indexer"):
            self.logger.warning("Querying against an empty index")
            return

        traversal_paths = parameters.get(
            "traversal_paths", self.default_traversal_paths
        )
        top_k = parameters.get("top_k", self.default_top_k)

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._rii_index.search(
                q=doc.embedding,
                L=self.L,
                topk=top_k,
            )
            for idx, dist in zip(indices, dists):
                match = Document(id=self._doc_ids[idx], embedding=self._vecs[idx])
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
