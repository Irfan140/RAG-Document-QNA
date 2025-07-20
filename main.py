import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings (laptop gone wild while using it)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader



# ---------------------- Load environment variables ----------------------
load_dotenv()

# Set environment variables explicitly (in case Streamlit loses them)
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Optional: Show error in UI if key is missing
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found in .env file")
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in .env file")



# ---------------------- LLM Setup ----------------------
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")



# ---------------------- Prompt Template ----------------------
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")



# ---------------------- Function: Create Vector DB ----------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("documents")  # Folder name

        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]  # Just take first 50 files
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )




# ---------------------- UI ----------------------
st.title("📄 RAG Document Q&A")

user_prompt = st.text_input("🔍 Enter your query from the research paper")

if st.button("📦 Create Document Embeddings"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready!")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start

        st.subheader("💬 Answer")
        st.write(response['answer'])
        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        # Display the context documents
        with st.expander("📄 Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Document {i+1}**")
                st.write(doc.page_content)
                st.markdown("---")
