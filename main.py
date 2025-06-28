from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_ollama.llms import OllamaLLM

class State(Dict):
    messages: List[Dict[str, str]]

llm = OllamaLLM(model="llama3.2")
graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response})
    return {"messages": state["messages"]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    print("📥 Incoming request:", req.messages)
    state = {"messages": req.messages}
    
    try:
        for event in graph.stream(state):
            for value in event.values():
                print("🤖 Assistant replied:", value["messages"][-1]["content"])
                return {"messages": value["messages"]}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 🔥 This will show the full error in terminal
        return {"error": str(e)}
