
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from node.generate import generate
from node.retrieve import retrieve_document
from state import GraphState

load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_document)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()