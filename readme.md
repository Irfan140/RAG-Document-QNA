# 📄 RAG Document Q&A

This is a Streamlit-based application that enables you to query your PDF research documents using **RAG (Retrieval-Augmented Generation)** with **LangChain**, **FAISS**, **OpenAI**, and **Groq's LLaMA 3 model**.

## ✨ Features

- Upload and parse PDF documents from a local directory
- Split documents into manageable text chunks
- Generate embeddings using OpenAI
- Store vectors using FAISS vectorstore
- Retrieve the most relevant chunks based on a user query
- Answer questions using LLaMA 3 (via Groq API)
- Shows context documents used to form the answer
- Clean, simple UI built with Streamlit

## 🛠️ Tech Stack

- `Streamlit` – UI
- `LangChain` – LLM orchestration
- `FAISS` – Vector store
- `OpenAIEmbeddings` – Embedding model
- `Groq LLaMA3` – Language model

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-doc-qna.git
   cd rag-doc-qna
  
2. Set up a virtual environment and install dependencies:

    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
3. Create a .env file and add your API keys:
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
4. Place your PDF documents inside a folder named documents/ in the project root.

5. Run the app:
    ```bash
    streamlit run app.py
## ⚠️ Notes
- Avoid using OllamaEmbeddings in unstable environments (noted from experience).

- Only first 50 PDFs are processed for performance reasons (can be adjusted).

- Ensure your API keys are valid and usage limits are considered.

