import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 1. Load and prepare raw text data from CSV
df = pd.read_csv("scraped_output_metadata.csv")
raw_text = " ".join(df["text"].dropna().tolist())

# 2. Chunking text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
docs = text_splitter.create_documents([raw_text])

# 3. Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. In-memory vector database (no persistence)
vectorstore = Chroma.from_documents(docs, embedding)

# 5. Language model (LLM) from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 256}
)

# 6. RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# 7. Main interface function used in Streamlit
def generate_answer(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"
