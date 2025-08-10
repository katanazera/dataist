import chainlit as cl
from workflow import graph, config
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="")

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

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="LangChain Helper",
            icon="https://i.imgur.com/fe4hpXD.png/-/scale_crop/200x200/center/",
            markdown_description="Hello, im Datais AI, Im helping analyse data.",
            starters=[
                cl.Starter(
                    label="How do you work?",
                    message="How do you work?",
                ),
            ],
        )
    ]