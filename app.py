import os
import io
import re

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            background-color: #fafafa;
        }

        .source-box {
            padding: 12px;
            border-left: 4px solid #4CAF50;
            background-color: #f5f5f5;
            margin-bottom: 10px;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The requested Groq model
GROQ_MODEL ="openai/gpt-oss-120b""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------

def get_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Cloud → App Settings → Secrets."
        )
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


# ---------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    document.close()

    return pages


# ---------------------------------------------------------
# Clean text
# ---------------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Chunk document
# ---------------------------------------------------------

def create_chunks(pages):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end]

            if chunk.strip():

                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number
                    }
                )

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------
# Create FAISS index
# ---------------------------------------------------------

def create_faiss_index(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    # Normalize vectors so inner product behaves like
    # cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Search relevant chunks
# ---------------------------------------------------------

def search_documents(
    question,
    index,
    chunks,
    model,
    top_k=TOP_K
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        results.append(
            {
                "text": chunks[index_id]["text"],
                "page": chunks[index_id]["page"],
                "score": float(score)
            }
        )

    return results


# ---------------------------------------------------------
# Generate answer with Groq
# ---------------------------------------------------------

def generate_answer(question, retrieved_chunks):

    client = get_groq_client()

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE - Page {item['page']}

{item['text']}
"""
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using ONLY the HR policy
information provided in the context.

Rules:

1. Do not invent HR policies.
2. Do not make assumptions that are not supported by the document.
3. If the answer is not available in the provided policy,
   clearly say that the uploaded HR policy does not contain
   enough information to answer the question.
4. Give clear and professional answers.
5. When possible, mention the relevant policy page.
6. For sensitive HR matters, recommend contacting HR when
   the policy requires interpretation or a management decision.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

QUESTION:

{question}

Answer the question based strictly on the HR policy context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=2,
        max_value=10,
        value=5
    )

    st.markdown("---")

    st.markdown(
        """
        **Technology Stack**

        - Streamlit
        - PyMuPDF
        - Sentence Transformers
        - FAISS
        - Groq
        - `openai/gpt-oss-200`
        """
    )


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">👩‍💼 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload an HR policy PDF and ask questions about leave,
    working hours, attendance, benefits, conduct, procedures,
    and other company policies.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing HR policy..."
        ):

            # 1. Extract PDF
            pages = extract_text_from_pdf(
                uploaded_file
            )

            if not pages:

                st.error(
                    "No readable text was found in this PDF. "
                    "If it is a scanned PDF, OCR may be required."
                )

                st.stop()

            # 2. Create chunks
            chunks = create_chunks(pages)

            # 3. Load embedding model
            embedding_model = load_embedding_model()

            # 4. Create FAISS index
            index = create_faiss_index(
                chunks,
                embedding_model
            )

            # Store in session state
            st.session_state.chunks = chunks

            st.session_state.faiss_index = index

            st.session_state.document_name = (
                uploaded_file.name
            )

        st.success(
            f"✅ {uploaded_file.name} processed successfully!"
        )

        st.info(
            f"Created {len(st.session_state.chunks)} "
            "searchable document chunks."
        )


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

if st.session_state.faiss_index is not None:

    st.markdown("---")

    st.subheader("💬 Ask about your HR policy")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: How many annual leaves can an employee take?"
        )
    )

    ask_button = st.button(
        "🔎 Ask HR Policy Assistant",
        type="primary"
    )

    if ask_button and question.strip():

        embedding_model = load_embedding_model()

        with st.spinner(
            "Searching HR policy..."
        ):

            retrieved_chunks = search_documents(
                question,
                st.session_state.faiss_index,
                st.session_state.chunks,
                embedding_model,
                top_k=top_k
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks
            )

        st.markdown("### 🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📚 Sources")

        for i, source in enumerate(
            retrieved_chunks,
            start=1
        ):

            st.markdown(
                f"""
                <div class="source-box">

                <strong>Source {i} — Page {source['page']}</strong>

                <br><br>

                {source['text']}

                <br><br>

                Similarity: {source['score']:.3f}

                </div>
                """,
                unsafe_allow_html=True
            )

elif uploaded_file is None:

    st.info(
        "👆 Upload an HR policy PDF to begin."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "HR Policy Assistant • RAG + FAISS + Sentence Transformers + Groq"
)
