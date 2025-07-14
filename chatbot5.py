import os
import pickle
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from datetime import datetime

load_dotenv()

prompt = PromptTemplate.from_template("""
You are a friendly, intelligent AI assistant helping a university student understand academic concepts. You answer questions using the uploaded academic documents (notes, textbooks, guides) and previous conversation history.

Your goals:
- Answer the student’s question using only the context provided.
- Keep your tone clear, beginner-friendly, and engaging — like a good tutor.
- Use analogies, examples, or breakdowns only if they appear in the documents.
- For casual input (hi, thanks), respond politely.
- Never guess factual answers unless shown in context.

context:
{context}
chat_history:
{chat_history}
question:
{question}

Give a clear, helpful answer in markdown format:
""")

st.set_page_config(page_title="NOTESBOT", layout="wide")
st.title("TEACHER NOTES Q&A CHATBOT")
st.markdown("Upload PDFs, DOCX, PPTX, CSV or TXT files to ask academic questions.")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    if st.button("Clear Chat History", key="clear_btn"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("Regenerate Vectorstore", key="regen_btn"):
        if os.path.exists("vectorstore.pkl"):
            os.remove("vectorstore.pkl")
        st.success("Vectorstore will be regenerated on next question.")
        st.rerun()

with col3:
    if st.button("Export Chat", key="export_btn"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"chat_history_{now}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for role, msg in st.session_state.get("chat_history", []):
                f.write(f"{role.upper()}: {msg}\n\n")
        with open(file_path, "rb") as f:
            st.download_button("Download Chat (.txt)", data=f, file_name=file_path, mime="text/plain")

uploaded_files = st.file_uploader(
    "Upload files for this session (PDF, DOCX, PPTX, CSV, TXT)",
    type=["pdf", "docx", "pptx", "csv", "txt"],
    accept_multiple_files=True
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading Permanent Vectorstore...")
def load_permanent_vectorstore():
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            return pickle.load(f)
    else:
        all_docs = []
        for fname in os.listdir("docs"):
            fpath = os.path.join("docs", fname)
            ext = fname.lower().split(".")[-1]
            if ext == "pdf":
                loader = PyPDFLoader(fpath)
            elif ext == "docx":
                loader = UnstructuredWordDocumentLoader(fpath)
            elif ext == "pptx":
                loader = UnstructuredPowerPointLoader(fpath)
            elif ext == "csv":
                loader = CSVLoader(fpath)
            elif ext == "txt":
                loader = TextLoader(fpath)
            else:
                continue

            raw = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(raw)
            all_docs.extend(chunks)

        vs = FAISS.from_documents(all_docs, embedding)
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vs, f)
        return vs

permanent_vs = load_permanent_vectorstore()

temp_vs = None
temp_dir = None
if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    all_temp_chunks = []
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_ext == "pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext == "csv":
            loader = CSVLoader(file_path)
        elif file_ext == "txt":
            loader = TextLoader(file_path)
        else:
            continue

        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        all_temp_chunks.extend(chunks)

    if all_temp_chunks:
        temp_vs = FAISS.from_documents(all_temp_chunks, embedding)
        all_temp_docs = temp_vs.similarity_search(" ", k=1000)
        permanent_vs.add_documents(all_temp_docs)

retriever = permanent_vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

user_input = st.chat_input("Ask something about your uploaded or saved documents...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
        answer = response["answer"]

        sources = []
        for doc in response.get("source_documents", []):
            metadata = doc.metadata
            fname = metadata.get("source", "?").split("/")[-1]
            page = metadata.get("page", "?")
            sources.append(f"{fname} (Page {page})")

        if sources:
            answer += "\n\nSource(s): " + ", ".join(sources)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", answer))

if uploaded_files and temp_dir:
    shutil.rmtree(temp_dir)
