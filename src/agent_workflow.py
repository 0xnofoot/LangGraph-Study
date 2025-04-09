import os
import asyncio
import operator

from typing import Annotated, List, Tuple, TypedDict, Union, Literal
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START


os.environ["OPENAI_API_KEY"] = "gNkfaoiXTKVUAZEhtFhTze"
os.environ["OPENAI_API_BASE"] = "http://10.2.189.113:8080"

os.environ["DEEPSEEK_API_KEY"] = "sk-9a5a49236e2c4cc9b557c02e9ff8a091"

os.environ["TAVILY_API_KEY"] = "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"

tools = [TavilySearchResults(max_results=1)]

# prompt = hub.pull("wfh/react-agent-executor")
# print(type(prompt))
prompt = ChatPromptTemplate.from_template("""\
================================ System Message ================================

You are a helpful assistant.

============================= Messages Placeholder =============================

{messages}
""")

# llm = ChatOpenAI(model="gpt-4o")
llm = ChatDeepSeek(model="deepseek-chat")
agent_executor = create_react_agent(llm, tools, prompt=prompt)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """未来要执行的计划"""

    steps: List[str] = Field(description="需要执行的不同步骤，应该按顺序排列")


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
                对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。\
                不要添加任何多余的步骤。最后一步的结果应该是最终答案。\
                确保每一步都有所有必要的信息 - 不要跳过步骤。\
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

# planner = planner_prompt | ChatOpenAI(
#     model="gpt-4o", temperature=0
# ).with_structured_output(Plan)
planner = planner_prompt | ChatDeepSeek(
    model="deepseek-chat", temperature=0
).with_structured_output(Plan)


class Response(BaseModel):
    """用户响应"""

    response: str


class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan。"
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """\
        对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。\
        不要添加任何多余的步骤。最后一步的结果应该是最终答案。\
        确保每一步都有所有必要的信息 - 不要跳过步骤。\

        你的目标是:
        {input}

        你的原计划是:
        {plan}

        你目前已完成的步骤是：
        {past_steps}

        相应地更新你的计划。如果不需要更多步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分。\
    """
)

# replanner = replanner_prompt | ChatOpenAI(
#     model="gpt-4o", temperature=0
# ).with_structured_output(Act)
replanner = replanner_prompt | ChatDeepSeek(
    model="deepseek-chat", temperature=0
).with_structured_output(Act)


async def main():
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""对于以下计划:
{plan_str}\n\n你的任务是执行第{1}步，{task}。"""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )

        return {
            "past_steps": state["past_steps"]
            + [(task, agent_response["messages"][-1].content)]
        }

    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    workflow = StateGraph(PlanExecute)

    workflow.add_node("planner", plan_step)

    workflow.add_node("agent", execute_step)

    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    workflow.add_edge("planner", "agent")

    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
    )

    app = workflow.compile()

    graph_png = app.get_graph().draw_mermaid_png()

    with open("agent_workflow.png", "wb") as f:
        f.write(graph_png)

    config = {"recursion_limit": 50}

    inputs = {
        "input": "2024年巴黎安运会100米自由泳决赛冠军的家乡是哪里？给出具体的位置。请问中文答复。"
    }

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


asyncio.run(main())
