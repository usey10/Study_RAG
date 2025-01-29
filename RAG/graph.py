
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from node.generate import generate
from node.retrieve import retrieve_document
from node.rerank import rerank_docs
from state import GraphState

load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_document)
workflow.add_node("rerank", rerank_docs)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()