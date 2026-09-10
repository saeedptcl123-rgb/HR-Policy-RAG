# 👩‍💼 HR Policy Assistant

An AI-powered HR Policy Assistant built with **RAG (Retrieval-Augmented Generation)**.

Users can upload an HR policy PDF and ask questions about the contents of the policy.

The application retrieves the most relevant sections from the uploaded document using vector similarity search and sends those sections to a Groq-hosted language model to generate the answer.

## 🚀 Features

* Upload HR policy PDFs
* Extract text using PyMuPDF
* Split documents into chunks
* Generate embeddings using Sentence Transformers
* Store embeddings in FAISS
* Retrieve the most relevant policy sections
* Generate answers using Groq
* Uses `openai/gpt-oss-200`
* Shows source pages and retrieved text
* Streamlit web interface
* No external vector database required

## 🏗️ Architecture

```text
                 HR Policy PDF
                       │
                       ▼
                ┌─────────────┐
                │  PyMuPDF    │
                │ PDF Extract │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │  Chunking   │
                └──────┬──────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Sentence          │
             │ Transformers      │
             │ Embeddings        │
             └─────────┬─────────┘
                       │
                       ▼
                 ┌──────────┐
                 │  FAISS   │
                 │  Index   │
                 └────┬─────┘
                      │
             User Question
                      │
                      ▼
             Question Embedding
                      │
                      ▼
             FAISS Similarity Search
                      │
                      ▼
             Top Relevant Chunks
                      │
                      ▼
                 Groq API
                      │
                      ▼
          openai/gpt-oss-200
                      │
                      ▼
               HR Answer
```

## 🛠️ Technology Stack

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers
* FAISS
* NumPy
* OpenAI Python SDK
* Groq API
* `openai/gpt-oss-200`

## 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔑 Groq API Key

The application expects a Streamlit secret called:

```text
GROQ_API_KEY
```

Do not put your API key directly inside `app.py`.

For Streamlit Community Cloud, add the secret through the application's **Advanced Settings → Secrets** section.

Example:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

## ▶️ How the Application Works

### 1. Upload PDF

The user uploads an HR policy PDF.

### 2. Extract Text

PyMuPDF extracts readable text from every PDF page.

### 3. Chunking

The extracted text is divided into overlapping chunks.

### 4. Embeddings

Sentence Transformers converts each chunk into a numerical vector.

### 5. FAISS

FAISS stores the vectors and performs similarity search.

### 6. Retrieval

When the user asks a question, the question is converted into an embedding.

FAISS retrieves the most relevant chunks from the HR policy.

### 7. Generation

The retrieved chunks are sent to Groq.

The language model generates an answer using the retrieved policy information.

## ⚠️ Limitations

This application works best with text-based PDFs.

Scanned/image-only PDFs may require OCR before the application can retrieve their text.

The FAISS index is created in application memory for the uploaded document. It is not a permanent database.

The application should not be considered a substitute for professional HR, legal, or compliance advice.

## ☁️ Deployment

The application can be deployed using:

* GitHub
* Streamlit Community Cloud

No local Python environment is required for deployment.

## 🔐 Security

Never commit API keys to GitHub.

Use Streamlit Secrets for the Groq API key.

## 📜 License

This project is provided for educational and demonstration purposes.
