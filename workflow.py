from dotenv import load_dotenv
import chainlit as cl
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition,ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import tools
from prompts import dataist_agent_prompt,fields_analyzer_prompt

load_dotenv()

tools = tools
llm = ChatOpenAI(model='gpt-4o-mini')
llm_with_tools = llm.bind_tools(tools)

class State(MessagesState):
    summary: str
    fields: dict | None

def check_fields(state: State):
    data = cl.user_session.get('dataframe')
    fields = cl.user_session.get('fields')

    if data is not None and not data.empty and not fields:
        return 'analyze_fields'
    
    return 'assistant'

def analyze_fields(state: State):
    df = cl.user_session.get('dataframe')

    # Create prompt template
    template = ChatPromptTemplate.from_messages([
    ("system", fields_analyzer_prompt()),
    
    ("human", """Dataframe head:

            {dataframe_head}

            Rows: {rows_count}
            Columns: {columns_count}

            Analyze each column and return ONLY JSON with column descriptions.""")
])
    
    #prepare data
    dataframe_head = df.head().to_string()
    rows_count = len(df)
    columns_count = len(df.columns)
    
    #format prompt
    prompt = template.format_messages(
        dataframe_head=dataframe_head,
        rows_count=rows_count,
        columns_count=columns_count
    )
    
    #get response from LLM
    response = llm.invoke(prompt)
    
    #parse str type to dict
    fields_dict = json.loads(response.content.strip())
        
    #save to state
    state["fields"] = fields_dict
    
    #save to user session
    cl.user_session.set("fields", fields_dict)
    cl.user_session.set("should_send_file", True)

    return {"fields": fields_dict}

#building a system prompt
def build_system_prompt(state: State) -> str:
    sys_content = dataist_agent_prompt()

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
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
    return {'summary': respone.content,'messages':[]}

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
builder.add_node('analyze_fields',analyze_fields)

builder.add_conditional_edges(
    START,
    check_fields,
    {
        'analyze_fields': 'analyze_fields',
        'assistant': 'assistant'
    }
)
builder.add_conditional_edges(
    'assistant',
    tools_condition,
    {
        'tools':'tools',
        END:END
    }
)
builder.add_edge('analyze_fields','assistant')
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