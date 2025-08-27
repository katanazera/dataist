from dotenv import load_dotenv
import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition,ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import tools
import prompts

load_dotenv()

tools = tools
llm = ChatOpenAI(model='gpt-4o-mini')
llm_with_tools = llm.bind_tools(tools)

class State(MessagesState):
    summary: str
    
#building a system prompt
def build_system_prompt(state: State) -> str:
    sys_content = prompts.DATAIST_PROMPT

    description = cl.user_session.get("fields")
    if description:
        sys_content += f'\n\nDescription of fields for better understanding: {description}'

    summary = state.get('summary', '')
    if summary:
        sys_content += f'\n\nSummary of conversation earlier: {summary}'

    return sys_content

#check messages for safe summarization to not interrupt tool callings.
def is_safe_for_summarization(message):
    """Check if message is safe for summarization"""
    if hasattr(message, 'tool_calls') and message.tool_calls:
        return False
    if hasattr(message, 'tool_call_id') and message.tool_call_id:
        return False
    if not hasattr(message, 'content') or not message.content:
        return False
    return True

#define the logic to call llm
def assistant(state: State) -> dict:
    sys_msg = SystemMessage(content=build_system_prompt(state))
    messages = [sys_msg] + state['messages']

    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

def summarize_text(state: State):

    summary = state.get('summary', '')

    #summarization prompt
    if summary:
        summary_message = (
            f'This is summary of the conversation: {summary}\n\n'
            'Extend the summary by taking into account the new messages.'
        )
    else:
        summary_message = (
            '\n\nCreate a summary of this conversation.'
        )

    #add prompt to our history
    messages = state['messages'] + [HumanMessage(content=summary_message)]
    respone = llm.invoke(messages)

    #filter messages to recent 2 messages
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {'summary': respone.content}

#decides when we need to filter our messages history
def should_continue(state: State):
    """Return next node to execute"""
    
    messages = state["messages"]
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # check for final message of agent
    if not (isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None)):
        return END
    
    # check for real message not a tool call
    safe_messages = [msg for msg in messages if is_safe_for_summarization(msg)]
    
    if len(safe_messages) > 6:
        return "summarize_text"
    
    return END
    
#graph builder logic nodes and edges
builder = StateGraph(State)

builder.add_node('assistant',assistant)
builder.add_node('tools',ToolNode(tools))
builder.add_node(summarize_text)

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    tools_condition,
    {
        'tools':'tools',
        END:END
    }
)
builder.add_conditional_edges(
    'assistant',
    should_continue,
    {
        "summarize_text": "summarize_text",
        END: END
    }
)
builder.add_edge('tools','assistant')
builder.add_edge('summarize_text',END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': '123'}}

if __name__ == '__main__':
    with open("assets/graph_llm.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())
    print('graph saved')