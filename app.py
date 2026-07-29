import os
import tempfile
from io import BytesIO

import streamlit as st

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI


# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
background:linear-gradient(135deg,#111827,#0f172a,#1e293b);
}

.title{
font-size:45px;
font-weight:700;
text-align:center;
color:white;
margin-bottom:5px;
}

.subtitle{
text-align:center;
color:#cbd5e1;
margin-bottom:30px;
}

.block{
background:rgba(255,255,255,0.05);
padding:20px;
border-radius:18px;
border:1px solid rgba(255,255,255,0.08);
backdrop-filter: blur(14px);
}

.chat-user{
background:#2563eb;
padding:15px;
border-radius:15px;
color:white;
margin-bottom:10px;
}

.chat-ai{
background:#1e293b;
padding:15px;
border-radius:15px;
color:white;
margin-bottom:10px;
border-left:5px solid #7c3aed;
}

.metric{
background:#0f172a;
padding:15px;
border-radius:12px;
text-align:center;
margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------
# TITLE
# --------------------------------------------------------

st.markdown("<div class='title'>📚 AI Document Assistant</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Upload your PDF and chat with it using Retrieval-Augmented Generation (RAG)</div>",
unsafe_allow_html=True,
)

# --------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pages" not in st.session_state:
    st.session_state.pages = 0

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------

with st.sidebar:

    st.title("⚙ Settings")

    st.markdown("---")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("Document Stats")

    st.metric(
        "Pages",
        st.session_state.pages
    )

    st.metric(
        "Chunks",
        len(st.session_state.chunks)
    )

    st.markdown("---")

    st.info(
        "Powered by\n\nOpenRouter + LangChain + FAISS"
    )

# --------------------------------------------------------
# LOAD OPENROUTER
# --------------------------------------------------------

api_key = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --------------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------------

def extract_text(pdf):

    reader = PdfReader(pdf)

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text())

    return "\n".join(pages), len(reader.pages)


def build_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        chunks,
        embedding=embeddings,
    )

    return vectorstore, chunks


# --------------------------------------------------------
# PDF INGESTION
# --------------------------------------------------------

if uploaded_pdf is not None:

    with st.spinner("Reading PDF..."):

        text, total_pages = extract_text(uploaded_pdf)

        vectorstore, chunks = build_vectorstore(text)

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.session_state.pages = total_pages

    st.success("PDF indexed successfully!")

st.markdown("---")
# --------------------------------------------------------
# CHAT
# --------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------
# ASK QUESTION
# --------------------------------------------------------

question = st.chat_input("Ask something about your PDF...")

if question:

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching document..."):

            docs = st.session_state.vectorstore.similarity_search(
                question,
                k=4
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an AI assistant.

Answer ONLY using the information from the context.

If the answer cannot be found in the context, reply exactly:

"I couldn't find this information in the uploaded document."

------------------------
Context:

{context}

------------------------

Question:

{question}

Answer:
"""

            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
            )

            answer = response.choices[0].message.content

        placeholder = st.empty()

        streamed = ""

        for word in answer.split():

            streamed += word + " "

            placeholder.markdown(streamed + "▌")

        placeholder.markdown(streamed)

        with st.expander("Retrieved Context"):

            st.write(context)

        if "couldn't find" in answer.lower():

            st.warning(
                "The answer was not found in the uploaded document."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# --------------------------------------------------------
# DOWNLOAD CHAT
# --------------------------------------------------------

if len(st.session_state.messages):

    history = ""

    for msg in st.session_state.messages:

        history += f"{msg['role'].upper()}\n"

        history += msg["content"]

        history += "\n\n"

    st.download_button(
        "📥 Download Chat",
        history,
        file_name="chat_history.txt",
        mime="text/plain"
    )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------

st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit • LangChain • FAISS • OpenRouter"
)
