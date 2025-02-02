
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from node.generate import generate
from node.retrieve import ensemble_document
from node.multiquery import generate_transformed_queries
from node.multiqueryretreive import multiquery_retrieve
from node.querymerge import merge_results

from node.rerank import rerank_docs
from state import GraphState

load_dotenv()

workflow = StateGraph(GraphState)

workflow.add_node("ensemble retrieve", ensemble_document)
workflow.add_node("multi query generate", generate_transformed_queries)
workflow.add_node("multi query retreive", multiquery_retrieve)
workflow.add_node("merge retrieve", merge_results)
workflow.add_node("rerank", rerank_docs)
workflow.add_node("generate", generate)

workflow.add_edge(START, "ensemble retrieve")
workflow.add_edge(START, "multi query generate")
workflow.add_edge("multi query generate", "multi query retreive")

# 조건부 함수 정의
def check_conditions(state: GraphState):
    # ensemble, multi query Retreive Done 확인
    if "ensemble_context" in state and "multi_context" in state:
        return [Send("merge retrieve", state)]
    return [] # 조건 미충족 시 빈 리스트(대기)

workflow.add_conditional_edges(
    "ensemble retrieve", check_conditions, ["merge retrieve"]
)
workflow.add_conditional_edges(
    "multi query retreive", check_conditions, ["merge retrieve"]
)

workflow.add_edge("merge retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()