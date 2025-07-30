import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. Load the CSV
df = pd.read_csv("scraped_output_metadata.csv")
print("📄 CSV Loaded. Columns:", df.columns)

# 2. Combine text from the 'text' column
df["text"] = df["text"].fillna("")  # 혹시 null 값이 있으면 공백으로 대체
raw_text = " ".join(df["text"].tolist())
print("🔍 Raw text length:", len(raw_text))
print("🔍 Raw text sample:", raw_text[:300])

# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)
chunks = text_splitter.create_documents([raw_text])
print(f"✅ Successfully created {len(chunks)} text chunks!")

# 4. Embedding + Chroma Vectorstore 저장
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db_path = "chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=chroma_db_path
)
vectorstore.persist()
print(f"✅ Chroma vectorstore saved to '{chroma_db_path}'")
