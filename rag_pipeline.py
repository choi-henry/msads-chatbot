# rag_pipeline.py
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 1. Load and prepare data
df = pd.read_csv("scraped_output_metadata.csv")
raw_text = " ".join(df["text"].dropna().tolist())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = text_splitter.create_documents([raw_text])

# 2. Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. In-memory Chroma vectorstore
vectorstore = Chroma.from_documents(docs, embedding)

# 4. HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 256}
)

# 5. RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# 6. Streamlit interface function
def generate_answer(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"
