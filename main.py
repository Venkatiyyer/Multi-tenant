# --- Monkey-patch Python's sqlite3 to use pysqlite3's newer build ---
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import shutil
# import chromadb



# Load environment variables
def load_env():
    load_dotenv()
load_env()

groq_api_key = os.getenv("GROQ_API_KEY")

# --- Credentials ---
CREDENTIALS = {
    "companyA": {"uploader": "uploaderpwd", "venkat": "venkatpwd", "iyer": "iyerpwd"},
    "companyB": {"uploader": "uploaderpwd", "rajesh": "rajeshpwd", "kumar": "kumarpwd"}
}

# --- Paths & Models ---
PERSIST_DIR = Path("/tmp/vectorstores/shared_chroma")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"

# --- Session state init ---
for key in ["authenticated", "company_id", "username"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "authenticated" else ""

# --- Shared multitenant store loader using Chroma ---
def get_vectorstore():
    if "vectorstore" not in st.session_state:
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        st.session_state.vectorstore = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embedder,
            collection_name="multi_tenant_docs",
        )
    return st.session_state.vectorstore


# --- Pages ---
def page_login():
    st.title("Login")
    with st.form("login_form"):
        comp = st.text_input("Company ID (companyA/companyB)")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            c,u,p = comp.strip(), user.strip(), pwd.strip()
            if CREDENTIALS.get(c,{}).get(u) == p:
                st.session_state.update({"authenticated": True, "company_id": c, "username": u})
                st.success(f"Logged in as {u} for {c}")
            else:
                st.error("Invalid Company ID, Username or Password.")

def page_upload():
    st.title("Upload Documents")
    if not st.session_state.authenticated:
        st.warning("Please login first.")
        return
    if st.session_state.username != "uploader":
        st.error("Only uploader can upload files.")
        return

    # Truncate collection button - separate from file upload
    if st.button("🧹 Clear all data"):
        try:
            store = get_vectorstore()
            collection = store._client.get_collection(name="multi_tenant_docs")
            all_ids = collection.get(ids=None)["ids"]  # Fetch all document ids

            if all_ids:
                collection.delete(ids=all_ids)  # Delete all docs by id
                st.success("multi_tenant_docs collection truncated successfully.")
            else:
                st.info("Collection is already empty.")

            # Clear cached vectorstore in session_state if any
            if "vectorstore" in st.session_state:
                del st.session_state["vectorstore"]

        except Exception as e:
            st.error(f"Failed to truncate collection: {e}")

    

    
    
    
    
    
    file = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"])
    if not file:
        return

    comp = st.session_state.company_id
    data_dir = Path(f"data/{comp}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir/file.name
    path.write_bytes(file.read())
    st.success(f"Saved {file.name} for {comp}")

    # Load & split
    loader = PyPDFLoader(str(path)) if path.suffix == ".pdf" else TextLoader(str(path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{**chunk.metadata, "company": comp} for chunk in chunks]

    # Add to Chroma
    store = get_vectorstore()
    store.add_texts(texts=texts, metadatas=metadatas)
    store.persist()

    st.success("Indexed to shared Chroma store with metadata!")


def page_query():
    st.title("Query Documents")
    if not st.session_state.authenticated:
        st.warning("Please login first.")
        return
    if st.session_state.username == "uploader":
        st.info("Uploader cannot query.")
        return

    q = st.text_input("Enter your question:")
    if not q:
        return

    comp = st.session_state.company_id
    store = get_vectorstore()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": {"company": comp}}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    ans = qa.run(q)
    st.markdown(f"**Answer:** {ans}")


def main():
    st.set_page_config(page_title="Multi-Tenant RAG", layout="wide")
    
    # App-level Title and Header
    st.title("Multi-Tenant RAG")
    st.markdown("Company-specific Knowledge Retrieval")
    
    page = st.sidebar.radio("Navigate", ["Login","Upload","Query"])
    if page == "Login":
        page_login()
    elif page == "Upload":
        page_upload()
    else:
        page_query()

if __name__ == "__main__":
    main()
