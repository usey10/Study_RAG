from state import GraphState
from ingestion import retriever, ensemble_retriever, bm25_retriever

# def retrieve_document(state: GraphState) -> GraphState:
#     print("---RETRIEVE---")
#     questions = state["question"]
#     documents = retriever.invoke(questions)
#     print(documents)
#     return {"context": documents, "question": questions}

# ensemble retriever 로 변경
def retrieve_document(state: GraphState) -> GraphState:
    print("---RETRIEVE---")
    questions = state["question"]
    documents = ensemble_retriever.invoke(questions)
    print(documents)
    return {"context": documents, "question": questions}