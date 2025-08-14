import os, ast, re
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # 0.1.x OK
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# === Load CSV ===
base_dir = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(base_dir, "scraped_output_metadata.csv")
df = pd.read_csv(df_path)
if 'metadata' in df.columns:
    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {}))
else:
    df['metadata'] = [{}] * len(df)

# 텍스트 컬럼 자동 추론
TEXT_COL = "text" if "text" in df.columns else ("content" if "content" in df.columns else None)
if TEXT_COL is None:
    raise ValueError("CSV에 'text' 또는 'content' 컬럼이 필요합니다.")

# === Chunking ===
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text or "")

def build_chunks_dataframe(df: pd.DataFrame, chunk_size: int = 800, chunk_overlap: int = 80) -> pd.DataFrame:
    records = []
    for source_idx, row in df.iterrows():
        seg_chunks = chunk_text(row[TEXT_COL], chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(seg_chunks):
            meta = dict(row["metadata"] or {})
            meta["source_idx"] = source_idx
            meta["chunk_idx"] = chunk_idx
            records.append({"chunk_text": chunk, "metadata": meta})
    return pd.DataFrame(records)

df_chunks = build_chunks_dataframe(df)
all_chunks = df_chunks["chunk_text"].tolist()
all_metas = df_chunks["metadata"].tolist()

# === Embedding (미리 계산하지 말고 객체만 생성) ===
embedder: Embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 경량/안정
    encode_kwargs={"normalize_embeddings": True},
)

# === Build or Load FAISS ===
INDEX_DIR = os.path.join(base_dir, "faiss_index")
def load_or_build_faiss(texts, metas, embedder) -> FAISS:
    faiss_file = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(faiss_file):
        return FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)
    db.save_local(INDEX_DIR)
    return db

db: FAISS = load_or_build_faiss(all_chunks, all_metas, embedder)

# === Query Expansion ===
import re
def expand_query(query: str) -> List[str]:
    patterns = {
        r"\bcore courses?\b": ["required courses", "core curriculum", "mandatory courses"],
        r"\badmission requirements?\b": ["application requirements", "eligibility", "admission criteria"],
        r"\bcapstone\b": ["final project", "thesis", "culminating project"],
    }
    outs = {query}
    for pat, syns in patterns.items():
        if re.search(pat, query, flags=re.I):
            for s in syns:
                outs.add(re.sub(pat, s, query, flags=re.I))
    return list(outs)

# === Metadata Filter (FAISS는 filter 미지원 → 사후 필터) ===
def detect_metadata_filter(query: str) -> Optional[Dict[str, Any]]:
    q = query.lower()
    mapping = {
        "core course": {"section": {"$eq": "Core Courses"}},
        "elective": {"section": {"$eq": "Sample Elective Courses"}},
        "foundation": {"section": {"$eq": "Noncredit Courses"}},
        "capstone": {"section": {"$eq": "MS-ADS Capstone Sponsor Guide 2025"}},
        "application": {"super_section": {"$eq": "Master’s in Applied Data Science Application Requirements"}},
    }
    for trig, f in mapping.items():
        if trig in q:
            return f
    return None

def apply_meta_filter(docs, meta_filter: Optional[Dict[str, Any]]):
    if not meta_filter:
        return docs
    key, cond = next(iter(meta_filter.items()))
    if "$eq" in cond:
        target = cond["$eq"]
        return [d for d in docs if (d.metadata or {}).get(key) == target]
    return docs

# === RAG Prompt Creation ===
def create_rag_prompt(selected_query: str, db_retriever: FAISS, embedder: Embeddings, k_docs: int = 8) -> str:
    expanded_queries = expand_query(selected_query)
    meta_filter = detect_metadata_filter(selected_query)

    per_k = max(3, min(6, k_docs))
    retriever = db_retriever.as_retriever(search_type="mmr",
                                          search_kwargs={"k": per_k, "lambda_mult": 0.3})

    collected = []
    for exp_q in expanded_queries:
        hits = retriever.get_relevant_documents(exp_q)
        # FAISS는 filter 미지원 → 사후 필터링
        hits = apply_meta_filter(hits, meta_filter)
        collected.extend(hits)

    # ---- 빈 결과 폴백 1: 메타필터 해제하고 한 번 더 ----
    if not collected:
        for exp_q in expanded_queries:
            collected.extend(retriever.get_relevant_documents(exp_q))

    # ---- 빈 결과 폴백 2: 최소 프롬프트 반환 ----
    if not collected:
        return f"""You are an educational assistant.
No context was retrieved. Answer briefly based on general MSADS information style (do not hallucinate; say you don't have the info if unsure).

[QUESTION]
{selected_query}
"""

    # 중복 제거
    seen, uniq_docs = set(), []
    for d in collected:
        key = ((d.metadata or {}).get("source_idx"), (d.metadata or {}).get("chunk_idx"))
        if key not in seen:
            seen.add(key)
            uniq_docs.append(d)

    # === 안전한 벡터화(2D 보장) ===
    query_vec = np.asarray(embedder.embed_query(selected_query), dtype=np.float32).reshape(1, -1)
    doc_vecs_list = embedder.embed_documents([d.page_content for d in uniq_docs])
    doc_vecs = np.asarray(doc_vecs_list, dtype=np.float32)
    if doc_vecs.ndim == 1:
        doc_vecs = doc_vecs.reshape(1, -1)

    sim = cosine_similarity(query_vec, doc_vecs)[0]

    # 가벼운 키워드 부스트
    q_words = set(re.findall(r"\b\w+\b", selected_query.lower()))
    scored = []
    for d, s in zip(uniq_docs, sim):
        d_words = set(re.findall(r"\b\w+\b", d.page_content.lower()))
        scored.append((d, s + 0.05 * len(q_words & d_words)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_context = "\n".join([d.page_content for d, _ in scored[:k_docs]])

    return f"""You are an educational assistant.
Use the following context to answer the question about the MSADS program.

[CONTEXT]
{top_context}

[QUESTION]
{selected_query}

Answer clearly using short paragraphs and/or bullet points.
"""

# === HuggingFace Model ===
MODELS = {
    "1": {"model_name": "google/flan-t5-base", "model_type": "encoder-decoder"},
    # 무거운 모델은 옵션으로만 (CPU 환경 주의)
    # "2": {"model_name":"mistralai/Mistral-7B-Instruct-v0.3","model_type":"decoder only"},
    # "3": {"model_name":"meta-llama/Llama-2-7b-hf","model_type":"decoder only"},
}
MODEL_CONFIG = {
    "encoder-decoder": {"model_cls": "text2text-generation", "pipeline": AutoModelForSeq2SeqLM},
    "decoder only": {"model_cls": "text-generation", "pipeline": AutoModelForCausalLM},
}

def load_llm_model(selected_model: dict, max_tokens: int = 512):
    model_name = selected_model["model_name"]
    model_type = selected_model["model_type"]
    model_cls = MODEL_CONFIG[model_type]["model_cls"]
    model_pipe = MODEL_CONFIG[model_type]["pipeline"]

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = model_pipe.from_pretrained(model_name)  # 장비에 맞게 옵션 추가 가능
    pipe = pipeline(model_cls, model=mdl, tokenizer=tok, max_new_tokens=max_tokens)
    return HuggingFacePipeline(pipeline=pipe)

# === Entry ===
def generate_answer(question: str, model_choice: str = "1") -> str:
    llm = load_llm_model(MODELS[model_choice])
    rag_prompt = create_rag_prompt(question, db_retriever=db, embedder=embedder, k_docs=8)
    return llm(rag_prompt)

