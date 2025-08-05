from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from tools import tools
import prompts
from IPython.display import display

load_dotenv()


tools = tools
llm = ChatOpenAI(model='gpt-4o')
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content=prompts.DATAIST_PROMPT)

class State(MessagesState):
    file_path: str

def assistant(state: State) -> dict:
    return {'messages': [llm_with_tools.invoke([sys_msg] + state['messages'])]}


builder = StateGraph(State)

builder.add_node('assistant',assistant)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START, 'assistant')

builder.add_conditional_edges(
    'assistant',
    tools_condition,
)
builder.add_edge('tools','assistant')

react_graph = builder.compile()

query = [HumanMessage(content='What is correlation between death/s and age?')]
query = react_graph.invoke({'messages': query})

for m in query['messages']:
    m.pretty_print()