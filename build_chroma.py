# build_chroma.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. CSV file loading
df = pd.read_csv("scraped_output_metadata.csv")
print(" text preview:\n", df["text"].head())

# 2. Combine all text into a single string
raw_text = " ".join(df["text"].dropna().tolist())
print(" Raw text length:", len(raw_text))
print(" Raw text sample:", raw_text[:300])

# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)
chunks = text_splitter.create_documents([raw_text])
print(f" Successfully created {len(chunks)} text chunks!")

# 4. Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Chroma DB creation
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

db.persist()
print(" ChromaDB saved! (./chroma_db)")
