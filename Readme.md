# 📚 Personal Notes Q&A Bot (Streamlit + Gemini + FAISS)

This is a Streamlit app that allows you to upload your personal notes as a `.txt` file, visualize the embedded chunks using PCA, and ask questions about them using a Gemini-powered Retrieval-Augmented Generation (RAG) pipeline.

🔗 **Live Demo**:  
👉 [https://mswetha1708-personal-notes-bot.streamlit.app](https://mswetha1708-personal-notes-bot.streamlit.app)

This Streamlit app allows you to upload your personal notes as a `.txt` file, visualize the embedded chunks using PCA, and ask questions about them using a Gemini-powered Retrieval-Augmented Generation (RAG) pipeline.

---
---

## 🚀 Features

- 📝 Upload `.txt` notes
- 🔍 Automatic text chunking
- 🧠 Embedding with **Google Gemini Embedding API**
- 📊 2D PCA visualization of chunk embeddings
- 💬 Ask questions using **Gemini LLM** (via RetrievalQA)
- ⚡ Fast vector search using **FAISS**

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS (Facebook AI)](https://github.com/facebookresearch/faiss)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [scikit-learn](https://scikit-learn.org/) (for PCA visualization)

---

## 📦 Installation

### 1. Clone this repo

```bash
git clone https://github.com/your-username/personal-notes-bot.git
cd personal-notes-bot
```
### 2. Install requirements
```bash
pip install -r requirements.txt
```
### 3. Set up API key
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
### 4. Run the app
```bash
streamlit run app.py
```