import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]

st.title("📚 Ask Your Notes + Visualize Embeddings (Gemini + FAISS)")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

def visualize_faiss_embeddings(faiss_store, num_points=10):
    docs = list(faiss_store.docstore._dict.values())
    if len(docs) < 2:
        st.warning("Need at least 2 chunks to visualize embeddings.")
        return

    vectors = [faiss_store.index.reconstruct(i) for i in range(min(len(docs), num_points))]
    texts = [doc.page_content for doc in docs[:num_points]]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(reduced[:, 0], reduced[:, 1], color='skyblue')
    for i, txt in enumerate(texts):
        snippet = txt[:30].replace("\n", " ") + "..."
        ax.annotate(snippet, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    ax.set_title("📉 FAISS Embedding Visualization (PCA)")
    ax.grid(True)
    st.pyplot(fig)

if uploaded_file:
    notes = uploaded_file.read().decode("utf-8")
    st.text_area("Preview of your notes:", notes[:1000], height=200)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = splitter.create_documents([notes])
    st.success(f"✅ Text split into {len(documents)} chunks.")

    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Visualization
    st.subheader("📉 Embedding Visualization (2D PCA)")
    try:
        visualize_faiss_embeddings(db, num_points=20)
    except Exception as e:
        st.error(f"Embedding visualization failed: {e}")

    # Q&A Interface
    st.subheader("💬 Ask a question based on your notes")
    query = st.text_input("Your question:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.success("Answer:")
            st.write(answer)