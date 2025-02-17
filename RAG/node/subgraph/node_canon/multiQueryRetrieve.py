# multiQueryretrieve.py
# not use
from state import GraphState
from node.subgraph.ingestion import retriever
from node.subgraph.node_canon.multiQuery import llm_chain

from langchain.retrievers.multi_query import MultiQueryRetriever

# Initialize the retriever
multiquery_retriever = MultiQueryRetriever(
    retriever=retriever, llm_chain=llm_chain, parser_key="lines"
)

# Node for retrieving documents
def multiquery_retrieve(state: GraphState) -> GraphState:
    print("---QUERY RETRIEVE---")
    transformed_queries = state["transform_question"]
    unique_docs = multiquery_retriever.invoke(transformed_queries)
    # print(unique_docs)
    return {"multi_context": unique_docs}