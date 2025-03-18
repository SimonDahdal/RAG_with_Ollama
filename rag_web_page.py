import gradio as gr
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:1b"

def call_ollama_model(question, context):
    """Call the Ollama model with a question and its context."""
    prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def rag_chain(url, question):
    """Load the web page from the given URL, retrieve context and generate answer using the LLM."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    combined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return call_ollama_model(question, combined_context)

def get_important_facts(url, question):
    """Return an answer to the user's question given a web page URL."""
    return rag_chain(url, question)

# Gradio interface with two inputs: one for the URL and one for the question
iface = gr.Interface(
    fn=get_important_facts,
    inputs=[
        gr.Textbox(lines=1, placeholder="Enter the web page URL here...", label="Web Page URL"),
        gr.Textbox(lines=1, placeholder="Enter your question here...", label="Question")
    ],
    outputs="text",
    title="RAG with Gemma3:1b",
    description="Enter a web page URL and ask questions about the page."
)

if __name__ == "__main__":
    iface.launch()