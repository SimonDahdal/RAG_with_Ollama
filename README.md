# RAG-Based PDF and Web Q&A

This project provides two main approaches for question-answering:

1. **`rag_pdf.py`**  
   Reads and processes a PDF, embeds the text in a vector store, retrieves relevant snippets, and uses a language model to answer questions.

2. **`rag_web_page.py`**  
   Similar approach but loads data from a web page instead of a PDF.

## How to Use

1. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
2. Run the PDF Q&A script:  
   ```
   python rag_pdf.py
   ```
   You will be prompted for your question.
3. Launch the web Q&A interface:  
   ```
   python rag_web_page.py
   ```
   Opens a Gradio UI where you can enter a URL and a question.

Feel free to customize the embeddings, chunk sizes, and language model settings in the scripts.
