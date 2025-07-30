# rag_pipeline.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Load .env (for HuggingFace token)
load_dotenv()

# 1. Load and chunk data
df = pd.read_csv("scraped_output_metadata.csv")
raw_text = " ".join(df["text"].dropna().tolist())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = text_splitter.create_documents([raw_text])

# 2. Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. In-memory vectorstore
vectorstore = Chroma.from_documents(docs, embedding)

# 4. HuggingFace model
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

# 6. Interface function
def generate_answer(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
