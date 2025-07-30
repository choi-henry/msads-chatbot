# rag_pipeline.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

# Hugging Face Hub 토큰 (필요한 경우 환경변수로 설정)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"  # TODO: 이 줄은 본인 토큰으로 바꾸기

# 1. Load embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load vector database (chroma DB must be already built in ./chroma_db)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# 3. Choose LLM from Hugging Face (you can change the model)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.2, "max_length": 256}
)

# 4. Create RAG-style QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# 5. Interface for Streamlit
def generate_answer(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
