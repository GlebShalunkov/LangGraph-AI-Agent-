from langchain_community.llms import Ollama

llm = Ollama(model="qwen3:4b")
response = llm.invoke("Привет!")
print(response)