import PyPDF2
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration settings for text chunking and models
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:1b"

def call_ollama_model(question, context):
    """
    Calls the Ollama language model with the provided question and context.
    Returns the answer text.
    """
    prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def load_pdf(pdf_path):
    """
    Opens a PDF file, extracts all text, and returns a list of simple Document-like objects.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    # Document-like class with page_content and metadata
    class Document:
        def __init__(self, page_content):
            self.page_content = page_content
            self.metadata = {}

    return [Document(text)]

def rag_chain(pdf_path, question):
    """
    Splits the PDF text, embeds the chunks into a vector store, retrieves the most relevant chunks,
    and then calls the language model to generate a final answer.
    
    1) load_pdf(pdf_path): Extracts text from the PDF.
    2) text_splitter: Splits the text into manageable chunks.
    3) OllamaEmbeddings: Creates embeddings for the chunks using the specified model.
    4) Chroma: Builds a vector store from the embedded chunks.
    5) as_retriever(): Prepares the vector store for question-answer retrieval.
    6) retrieve.invoke(question): Retrieves the most relevant chunks for the question.
    7) call_ollama_model: Passes the question and retrieved text to the language model for the final answer.
    """
    docs = load_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)
    combined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return call_ollama_model(question, combined_context)

def get_answer_from_pdf(pdf_path, question):
    """
    Main function to ingest a PDF path and a question, returning the model's answer.
    """
    return rag_chain(pdf_path, question)

if __name__ == "__main__":
    pdf_path = "<link_to_your_pdf_file.pdf>"
    query = input("Enter your question: ")
    answer = get_answer_from_pdf(pdf_path, query)
    print(answer)