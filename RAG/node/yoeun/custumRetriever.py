from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any, Callable, Dict, Iterable, List, Optional
from langchain_core.documents import Document


from langchain_community.retrievers import BM25Retriever

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

class FilteredBM25Retriever(BM25Retriever):
    model_filter: Optional[str] = None  # 특정 모델 필터링 추가

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        retrieved_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)

        # 특정 model 값이 있는 문서만 필터링
        if self.model_filter:
            retrieved_docs = [doc for doc in retrieved_docs if doc.metadata.get("model") == self.model_filter]

        return retrieved_docs