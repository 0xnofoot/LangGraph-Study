import os

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

os.environ["OPENAI_API_KEY"] = "gNkfaoiXTKVUAZEhtFhTze"
os.environ["OPENAI_API_BASE"] = "http://10.2.183.150:8080"
os.environ["TAVILY_API_KEY"] = "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"


@tool
def search(query: str):
    """模拟一个搜索工具"""
    if "上海" in query.lower() or "shanghai" in query.lower():
        return "现在30度，有雾。"

    return "现在35度，阳光明媚。"


tools = [search]

tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"

    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


workflow = StateGraph((MessagesState))

workflow.add_node("agent", call_model)

# ......
