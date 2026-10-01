[README (2).md](https://github.com/user-attachments/files/32895453/README.2.md)
# 🔍 InsightLense — Multimodal RAG Research Document Assistant

InsightLense is a multimodal AI-powered research document assistant that allows users to upload PDF documents and ask questions about their content using natural language.

The system combines document parsing, visual analysis, semantic retrieval, keyword retrieval, and large language models to answer questions about text, tables, charts, and diagrams.

🌐 **Live Demo:** https://insightlense-rag-research-assistant.onrender.com/

💻 **GitHub:** https://github.com/SiddharthaSangani18/InsightLense-RAG-Research-Assistant

---

## ✨ Features

- 📄 PDF document upload
- 🧠 Multimodal document understanding
- 👁️ Gemini Vision analysis for diagrams, charts, and visual content
- 🔎 Semantic search using vector embeddings
- 🔤 Keyword search using BM25
- ⚡ Hybrid retrieval using FAISS + BM25
- 🤖 AI-powered question answering using Groq
- 💬 Conversation history
- 🔄 Change Document functionality
- 🗑️ Session reset
- 📊 Support for questions about text, tables, charts, figures, and diagrams

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │        User         │
                    │  Upload PDF / Ask   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │    LlamaParse   │        │    PyMuPDF      │
        │  Text Extraction│        │ Visual Detection │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 │                           ▼
                 │                  ┌─────────────────┐
                 │                  │   Gemini Vision │
                 │                  │  Visual Analysis│
                 │                  └────────┬────────┘
                 │                           │
                 └────────────┬──────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Document Chunks   │
                    │ + Visual Analysis   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastEmbed      │
                    │   Vector Embeddings │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FAISS         │
                    │ Semantic Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       BM25          │
                    │ Keyword Retrieval   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Hybrid Retriever  │
                    │   FAISS + BM25      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    │  Answer Generation  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    AI Assistant     │
                    │      Response       │
                    └─────────────────────┘
```

---

## 🧠 How It Works

### 1. Upload PDF

The user uploads a research document through the web interface.

### 2. Extract Document Content

LlamaParse processes the PDF and extracts document content.

### 3. Analyze Visual Content

PyMuPDF detects images and drawings in PDF pages.

Pages containing visual content are rendered and analyzed using Gemini Vision. The visual analysis captures layout, positions, labels, charts, tables, diagrams, arrows, and relationships.

### 4. Create Document Chunks

The extracted text and visual analysis are combined and divided into chunks for retrieval.

### 5. Generate Embeddings

InsightLense uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through FastEmbed.

FastEmbed uses ONNX Runtime for lightweight embedding generation without requiring PyTorch.

### 6. Hybrid Retrieval

The application combines:

**FAISS**
- Semantic similarity search
- Finds conceptually related content

**BM25**
- Keyword-based retrieval
- Useful for exact terms, names, numbers, and phrases

These are combined into an ensemble retriever.

### 7. Generate the Answer

Relevant document chunks are passed to the Groq-powered LLM.

For questions about figures and diagrams, the system gives priority to the generated visual-analysis sections.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend API |
| LlamaParse | PDF/document parsing |
| PyMuPDF | PDF processing and visual detection |
| Gemini Vision | Visual understanding |
| FastEmbed | Embedding generation |
| ONNX Runtime | Lightweight embedding inference |
| FAISS | Vector similarity search |
| BM25 | Keyword retrieval |
| LangChain | RAG pipeline orchestration |
| Groq | LLM inference |
| HTML/CSS/JavaScript | Frontend |
| Render | Deployment |

---

## 📂 Project Structure

```text
InsightLense-RAG-Research-Assistant/
│
├── main.py
├── requirements.txt
├── .gitignore
│
└── static/
    └── index.html
```

Runtime-generated directories such as `vectorstore/` and `model_cache/` are excluded from Git.

---

## 🔑 Environment Variables

Create a `.env` file locally:

```env
GEMINI_API_KEY=your_gemini_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SiddharthaSangani18/InsightLense-RAG-Research-Assistant.git
cd InsightLense-RAG-Research-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create `.env` with the required API keys.

### 6. Start the application

```bash
python -m uvicorn main:app
```

The application will be available at:

```text
http://127.0.0.1:8000
```

---

## ☁️ Deployment

InsightLense is deployed on Render.

### Build Command

```bash
pip install -r requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='model_cache')"
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Python Version

```text
3.12.3
```

---

## 🔄 Session Management

Each uploaded document receives its own session.

The active FAISS vector store and retriever are kept in memory. When the user selects **Change Document**, the current session is cleared and a new document can be uploaded.

This prevents the previous document from being used when answering questions about a new document.

---

## 💡 Example Questions

After uploading a PDF, users can ask:

```text
What is this document about?

Summarize the main findings.

What does Figure 1.3 represent?

What is present at the center of the diagram?

What are the key values in the table?

What percentage is mentioned in the document?

Explain the relationship between these two concepts.

What are the main conclusions?
```

---

## 🔐 Security

- Store API keys as environment variables.
- Never commit `.env` to GitHub.
- Treat uploaded documents as potentially sensitive.
- For production use with private documents, add appropriate authentication and access controls.

---

## 🚧 Limitations

- Gemini Vision availability depends on the external Gemini API.
- Large PDFs can require more processing time and memory.
- Render Free has limited memory resources.
- The embedding model must be downloaded when it is not already cached.
- The current application focuses on PDF research document analysis.

---

## 🔮 Future Improvements

- 🔐 User authentication
- 📚 Multiple-document collections
- 🗂️ Persistent document management
- 📑 Source citations for answers
- 💾 Persistent vector database
- ⚡ Streaming AI responses
- 🧠 Improved multimodal retrieval
- 📄 Support for additional document formats
- ☁️ Cloud storage for uploaded documents
- 📊 Document analytics dashboard

---

## 👨‍💻 Author

**Siddhartha Sangani**

B.Tech — Computer Science & Engineering (AI & ML)

Hyderabad, India

### Project Links

🌐 **Live Demo:** https://insightlense-rag-research-assistant.onrender.com/

💻 **GitHub:** https://github.com/SiddharthaSangani18/InsightLense-RAG-Research-Assistant

---

## ⭐ Project

**InsightLense — Multimodal RAG Research Document Assistant**
