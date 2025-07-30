# rag_pipeline.py
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Build Chroma DB if not exists
if not os.path.exists("chroma_db"):
    os.system("python build_chroma.py")

# ✅ LangChain / HuggingFace import
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 1. Load embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load Chroma DB
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# 3. Load LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 256}
)

# 4. Setup RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# 5. Interface
def generate_answer(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
