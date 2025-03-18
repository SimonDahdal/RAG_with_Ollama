from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma3:1b", max_tokens=100)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"question": "What is LangChain?"}))