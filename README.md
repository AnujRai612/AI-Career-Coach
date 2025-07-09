# AI Career Coach - Resume Analyzer & Expert Q\&A

An AI-powered web application that analyzes resumes and answers career-related questions using Google Gemini Pro and LangChain.

---

## 🔍 Features

* ✉️ Upload a resume in PDF format
* 🤖 Extract key career highlights using LLM (Gemini)
* 🔎 Ask questions about your resume in natural language
* ⚖️ Get instant suggestions and summaries
* 🌍 Run locally with free Gemini Pro API

---

## 📊 Project Architecture

```
Frontend (HTML + TailwindCSS + JS)
       |
       | (Flask routes)
       v
Backend (Python Flask + LangChain + Google Gemini Pro)
       |
       ├── PDF Parsing (PyPDF2)
       ├── Resume Analysis using LangChain + Gemini
       ├── Embedding & Vector Store (FAISS + Sentence Transformers)
       └── QA over resume content
```

---

## 📂 Folder Structure

```
.
├── app.py                     # Main Flask app
├── templates/
│   ├── index.html             # Upload page
│   ├── results.html           # Resume summary output
│   ├── ask.html               # Ask a question
│   ├── qa_results.html        # QA result display
│   └── query_results.html     # (Optional additional view)
├── uploads/                   # Uploaded resume files
├── vector_index/             # FAISS index and vector store files
├── requirements.txt          # All dependencies
```

---

## 🚀 Tech Stack

* **Frontend:** HTML5, TailwindCSS, Framer Motion
* **Backend:** Python, Flask
* **PDF Parsing:** PyPDF2
* **LLM Integration:** Google Gemini Pro via LangChain
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
* **Vector Store:** FAISS
* **Hosting:** Localhost / Flask development server

---

## ⚙️ How It Works

1. **User uploads PDF resume**
2. **Backend parses text** using `PyPDF2`
3. Text is sent to:

   * **LLMChain** for career insights
   * **HuggingFaceEmbeddings** for vector search
4. User can ask a question

   * Search handled via **RetrievalQA** using **FAISS**
5. Results shown on `qa_results.html`

---

## 📅 Resume Content Example

**Project: AI Career Coach Web App**
*Role: Full Stack Developer*
*Technologies: Python, Flask, LangChain, Gemini Pro, FAISS, HuggingFace*

**Description:**

> Developed an AI-powered career coaching web app that extracts insights from resumes and allows users to ask questions powered by Google Gemini Pro. Integrated semantic search with FAISS and sentence-transformers to enable dynamic Q\&A over user resumes.

**Key Highlights:**

* Extracted data using PyPDF2 and embedded content for vector retrieval
* Used LangChain LLMChain to prompt Gemini for personalized responses
* Designed frontend with TailwindCSS and Flask routing
* Built vector database using HuggingFaceEmbeddings + FAISS
* Enabled custom question-answering pipeline over resume content

**Outcome:**

* Built a working MVP with end-to-end AI integration
* Reduced manual resume review by 80%

---

## 🚫 Limitations

* Gemini API key must be active and valid
* Currently runs on localhost (not deployed)
* Meant for demo/learning purposes (not production hardened)

---

## 🛌 How to Run

```bash
# Step 1: Clone and create environment
conda create -n ai-coach python=3.10 -y
conda activate ai-coach

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
python app.py

# Visit http://127.0.0.1:5000 in your browser
```

Make sure your `GOOGLE_API_KEY` is valid and set properly in the code before launching.

---

## 🌍 Credits

Built by Anuj Rai as part of an AI career transition portfolio project. Uses OpenAI, LangChain, Google Gemini Pro and open-source vector tech.
