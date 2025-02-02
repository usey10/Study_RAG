from state import GraphState

from sentence_transformers import SentenceTransformer
import numpy as np

# ingestion.py 에 추가 필요
model = SentenceTransformer("jhgan/ko-sbert-sts")


# 특정 키워드 기반 Percentile 필터링 함수
def filter_documents_by_percentile(docs_with_scores, percentile_cutoff=80):
    scores = [score for _, score in docs_with_scores]

    if not scores:
        return []

    cutoff_value = np.percentile(scores, percentile_cutoff)  # 상위 percentile 계산
    return [doc for doc, score in docs_with_scores if score >= cutoff_value]

def assign_embedding_similarity_score(docs, query_embedding):
    doc_embeddings = [model.encode(doc.page_content) for doc in docs]
    
    # Dot Product로 유사도 점수 계산
    scores = [np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]

    return list(zip(docs, scores))


def filter_document(state: GraphState) -> GraphState:
    # Query 및 문서 임베딩 생성
    query = state['question']
    docs = state['merge_context']

    query_embedding = model.encode(query)
    # Embedding 기반 유사도 점수 계산
    docs_with_scores = assign_embedding_similarity_score(docs, query_embedding)
    # Percentile Cutoff 적용 (상위 80%)
    filtered_docs = filter_documents_by_percentile(docs_with_scores, percentile_cutoff=80)

    return {"filtered_documents": filtered_docs}