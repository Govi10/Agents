from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    """State agent for the state graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the docuemnt with provided content."""
    global document_content
    document_content = content
    return f"Document updated with content: {document_content}"

@tool
def save(filename: str) -> str:
    """Save the document to a file.
    Args: 
    Filename: Name for the text file.
    """
    global document_content
           
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        return f"Document saved to {filename}"
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
    
tools = [update, save]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are my Drafter, You are going to help the user update and modify the document
        
        - If the user want to update the document, Use the udpate tool to update the document with the provided content.
        - If the user want to save the document, Use the save tool to save the document to a file.
        - Make sure to always show the current document state after modification.
        The current document state is: {document_content}
        """)
    
    if not state["messages"]:
        user_input = "I'm ready to help ypu update a document. What would you like to create"
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input = input(f" what would like to update in the document \n")
        print("\n USER : {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_message = [system_prompt] + list(state["messages"]+[user_message])
    
    response = model.invoke(all_message)
    
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}
    
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""
    
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for messages in reversed(messages):
        if(isinstance(messages, ToolMessage) and 
           "saved" in messages.content.lower() and
           "document" in messages.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
def tool_handler(state: AgentState) -> AgentState:
    # This function will handle tool calls if needed
    return state

graph.add_node("tools", tool_handler)

graph.set_entry_point("agent")

graph.add_edge("agent","tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue" : "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()

