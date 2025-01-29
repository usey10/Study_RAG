from state import GraphState
from ingestion import retriever

def retrieve_document(state: GraphState) -> GraphState:
    print("---RETRIEVE---")
    questions = state["question"]
    documents = retriever.invoke(questions)
    print(documents)
    return {"context": documents, "question": questions}