import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load environment variables
import streamlit as st
api_key = st.secrets["GOOGLE_API_KEY"]

st.title("📚 Ask Your Notes + Visualize Embeddings (Gemini)")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    notes = uploaded_file.read().decode("utf-8")
    st.text_area("Preview of your notes:", notes[:1000], height=200)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = splitter.create_documents([notes])
    st.success(f"✅ Text split into {len(documents)} chunks.")

    # Embed using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    chroma_settings = ChromaSettings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None  # <- disables disk writes
    )

    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        client_settings=chroma_settings
    )
    retriever = db.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Visualization
    st.subheader("📉 Embedding Visualization (2D PCA)")

    try:
        collection = db._collection
        raw = collection.get(include=["embeddings", "documents"])
        if len(raw["embeddings"]) < 2:
            st.warning("Need at least 2 chunks to visualize embeddings.")
        else:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(raw["embeddings"])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(reduced[:, 0], reduced[:, 1])
            for i, txt in enumerate(raw["documents"][:20]):
                ax.annotate(txt[:30].replace("\n", " ") + "...", (reduced[i, 0], reduced[i, 1]), fontsize=8)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Embedding visualization failed: {e}")

    # Ask question
    st.subheader("💬 Ask a question based on your notes")
    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.success("Answer:")
            st.write(answer)
