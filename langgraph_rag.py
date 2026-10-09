import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Загружаем базу знаний
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectordb = Chroma(persist_directory="./vectordb", embedding_function=embeddings)
    RAG_AVAILABLE = True
except Exception as e:
    print(f"RAG недоступен: {e}")
    RAG_AVAILABLE = False

# 1. Узел — ищет в документах
def search_docs(state: MessagesState):
    if not RAG_AVAILABLE:
        return {"messages": [AIMessage(content="RAG недоступен")]}

    query = state["messages"][-1].content
    docs = vectordb.similarity_search(query, k=3)

    if not docs:
        return {"messages": [AIMessage(content="Документы не найдены")]}

    context = "\n\n".join([doc.page_content for doc in docs])
    return {"messages": [AIMessage(content=f"Найдено:\n{context}")]}

# 2. Узел — отвечает на основе найденного
def call_model(state: MessagesState):
    # Собираем последнее сообщение (это контекст из поиска)
    last_message = state["messages"][-1].content
    
    prompt = f"""Ты — полезный ассистент. Ответь на вопрос, используя ТОЛЬКО следующий контекст.
Если в контексте есть точный ответ — дай его.
Если информации нет — скажи "В документах нет информации". НЕ ВЫДУМЫВАЙ.

Контекст:
{last_message}

Ответ:"""
    
    llm = Ollama(model="qwen3:4b", temperature=0.1)
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response)]}

# 3. Создаём граф
builder = StateGraph(MessagesState)

builder.add_node("search", search_docs)
builder.add_node("agent", call_model)

builder.add_edge(START, "search")
builder.add_edge("search", "agent")
builder.add_edge("agent", END)

graph = builder.compile()

# 4. Запускаем
print("Запуск графа...")
while True:
    question = input("\nЗадайте вопрос (или 'выход'): ")
    if question.lower() in ["выход", "exit", "quit"]:
        print("До свидания!")
        break
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    print("\nОтвет:")
    print(result["messages"][-1].content)