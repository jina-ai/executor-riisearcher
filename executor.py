from jina import Executor, DocumentArray, requests


class RiiSearcher(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
