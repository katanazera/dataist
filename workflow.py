from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
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

memory = MemorySaver()
react_graph = builder.compile(checkpointer=memory)

config = {'configurable': {'thread_id': '123'}}

query = [HumanMessage(content='Whats average age in data. data/titanic.csv, data/description.json')]
query = react_graph.invoke({'messages': query}, config)
query = [HumanMessage(content='How parameter i asked correlates with death ratio')]
query = react_graph.invoke({'messages': query}, config)
query = [HumanMessage(content='Do you know description of first 2 column, did you get it from file?')]
query = react_graph.invoke({'messages': query}, config)

for m in query['messages']:
    m.pretty_print()

if __name__ == '__main__':
    with open("assets/graph_llm.png", "wb") as f:
        f.write(react_graph.get_graph(xray=True).draw_mermaid_png())
    print('graph saved')