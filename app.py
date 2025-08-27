import chainlit as cl
import json
import pandas as pd
from workflow import graph, config
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_message
async def on_message(msg: cl.Message):
    user_session_id = cl.user_session.get("id")
    thanks_action = cl.Action(
        label="😎👍",
        name="thanks_action",
        payload={"user_session_id": user_session_id},
        tooltip="Send thanks for the helpful reply"
    )

    if msg.elements:
        for element in msg.elements:
            if element.name.lower().endswith(".csv"):
                df = pd.read_csv(element.path)
                cl.user_session.set("dataframe", df)
                await cl.Message(content=f"csv loaded: {element.name}, {df.shape[0]} lines").send()

            if element.name.lower().endswith(".json"):
                with open(element.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cl.user_session.set("fields", data)
                await cl.Message(content=f"json loaded: {element.name}, {len(data)} elements").send()

    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="",actions=[thanks_action])

    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(**config)
    ):
        if (
            isinstance(chunk, AIMessage)
            and metadata.get("langgraph_node") == "assistant"
            and not chunk.tool_calls
        ):
            if isinstance(chunk.content, str):
                await final_answer.stream_token(chunk.content)
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if block.get("type") == "text":
                        await final_answer.stream_token(block["text"])
    await final_answer.send()                    

@cl.action_callback("thanks_action")
async def on_action(action: cl.Action):
    print("message id:", action.forId, "action payload:", action.payload)
    await action.remove()
    await cl.Message(content="You are awesome <3").send()

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="LangChain Helper",
            icon="https://i.imgur.com/fe4hpXD.png/-/scale_crop/200x200/center/",
            markdown_description="Hello, im Dataist AI, Im helping analyze data.",
            starters=[
                cl.Starter(
                    label="How do you work?",
                    message="How do you work?",
                ),
            ],
        )
    ]