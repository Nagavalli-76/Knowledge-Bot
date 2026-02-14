import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

st.header("🤖 KnowledgeBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload PDF", type="pdf")

# Load FREE models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)


if file is not None:
    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    # Split text
    #chunks = [text[i:i+300] for i in range(0, len(text), 250)]
    chunks = [text[i:i + 500] for i in range(0, len(text), 450)]

    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    query = st.text_input("Ask a question")

    if query:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = index.search(query_embedding, k=3)

        if distances[0][0] > 0.8:
            st.write("❌ Answer not found in the given PDF.")
        else:
            context = " ".join([chunks[i] for i in indices[0]])

            prompt = f"""
            Answer the question briefly in 3–4 lines using ONLY the context below.
            If the answer is not in the context, say "Answer not found in the given PDF."

            Context:
            {context}

            Question:
            {query}
            """

            result = qa_pipeline(prompt, max_length=120)
            st.write("### ✅ Answer:")
            st.write(result[0]["generated_text"])


