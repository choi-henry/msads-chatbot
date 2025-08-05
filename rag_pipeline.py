import os
import ast
import pandas as pd
from typing import List, Optional, Dict, Any
import re
from sklearn.metrics.pairwise import cosine_similarity

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# === Load CSV ===
base_dir = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(base_dir, "scraped_output_metadata.csv")
df = pd.read_csv(df_path)
df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# === Chunking ===
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def build_chunks_dataframe(df: pd.DataFrame, chunk_size: int = 800, chunk_overlap: int = 80) -> pd.DataFrame:
    records = []
    for source_idx, row in df.iterrows():
        seg_chunks = chunk_text(row['text'], chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(seg_chunks):
            meta = row['metadata'].copy()
            meta['source_idx'] = source_idx
            meta['chunk_idx'] = chunk_idx
            records.append({'chunk_text': chunk, 'metadata': meta})
    return pd.DataFrame(records)

df_chunks = build_chunks_dataframe(df)
all_chunks = df_chunks['chunk_text'].tolist()
all_metas = df_chunks['metadata'].tolist()

# === Embedding ===
def embed_chunk_docs(chunk_docs: List[str], model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2", normalize: bool = True):
    embedder = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": normalize})
    embeddings = embedder.embed_documents(chunk_docs)
    return embedder, embeddings

embedder, _ = embed_chunk_docs(all_chunks)

# === FAISS DB Build ===
def store_embed_in_db(texts: List[str], metas: List[dict], embedder: Optional[Embeddings] = None, persist_directory: Optional[str] = "faiss_index") -> FAISS:
    db = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)
    db.save_local(persist_directory)
    return db

db = store_embed_in_db(all_chunks, all_metas, embedder=embedder)

# === Query Expansion & Metadata Filters ===
def expand_query(query: str) -> List[str]:
    expansions = {
        "core courses": ["required courses", "core curriculum", "fundamental courses", "mandatory courses"],
        "admission requirements": ["application requirements", "eligibility", "admission criteria"],
        "capstone": ["final project", "thesis", "culminating project"]
    }
    expanded_queries = [query]
    query_lower = query.lower()
    for key, synonyms in expansions.items():
        if key in query_lower:
            for synonym in synonyms:
                expanded_queries.append(query.replace(key, synonym, 1))
    return expanded_queries

def detect_metadata_filter(query: str) -> Optional[Dict[str, Any]]:
    q = query.lower()
    mapping = {
        "core course":     {"section": {"$eq": "Core Courses"}},
        "elective":        {"section": {"$eq": "Sample Elective Courses"}},
        "foundation":      {"section": {"$eq": "Noncredit Courses"}},
        "capstone":        {"section": {"$eq": "MS-ADS Capstone Sponsor Guide 2025"}},
        "application":     {"super_section": {"$eq": "Master’s in Applied Data Science Application Requirements"}},
    }
    for trigger, filt in mapping.items():
        if trigger in q:
            return filt
    return None

# === RAG Prompt Creation ===
def create_rag_prompt(selected_query: str, db_retriever: VectorStoreRetriever, embedder: Embeddings, k_docs: int = 10) -> str:
    expanded_queries = expand_query(selected_query)
    metadata_filter = detect_metadata_filter(selected_query)

    all_retrieved_docs = []
    for exp_query in expanded_queries:
        retriever = db_retriever.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_docs, "lambda_mult": 0.3, **({"filter": metadata_filter} if metadata_filter else {})}
        )
        results = retriever.get_relevant_documents(f"query: {exp_query}")
        for doc in results:
            all_retrieved_docs.append((doc.page_content, exp_query))

    seen_docs = set()
    unique_docs = [(d, q) for d, q in all_retrieved_docs if d not in seen_docs and not seen_docs.add(d)]

    query_embed = embedder.embed_query(f"query: {selected_query}")
    doc_embeds = embedder.embed_documents([f"passage: {doc}" for doc, _ in unique_docs])
    similarity_scores = cosine_similarity([query_embed], doc_embeds)[0]

    boosted_docs = []
    query_keywords = set(re.findall(r'\b\w+\b', selected_query.lower()))
    for (doc, exp_query), score in zip(unique_docs, similarity_scores):
        doc_keywords = set(re.findall(r'\b\w+\b', doc.lower()))
        boosted_score = score + 0.05 * len(query_keywords.intersection(doc_keywords))
        boosted_docs.append((doc, boosted_score))
    boosted_docs.sort(key=lambda x: x[1], reverse=True)

    context = "\n".join([doc for doc, _ in boosted_docs[:k_docs]])

    return f"""
    You are an educational assistant.
    Use the following context to answer the question about the MSADS program.

    [CONTEXT]
    {context}

    [QUESTION]
    query: {selected_query}

    Summarize clearly using paragraphs and/or bullet points.
    """

# === HuggingFace Model ===
MODELS = {
    "1": {"model_name":"google/flan-t5-base", "model_type": "encoder-decoder"},
    "2": {"model_name":"mistralai/Mistral-7B-Instruct-v0.3", "model_type": "decoder only"},
    "3": {"model_name":"meta-llama/Llama-2-7b-hf", "model_type": "decoder only"}
}
MODEL_CONFIG = {
    "encoder-decoder": {"model_cls": "text2text-generation", "pipeline": AutoModelForSeq2SeqLM},
    "decoder only": {"model_cls": "text-generation", "pipeline": AutoModelForCausalLM}
}

def load_llm_model(selected_model: dict,
                   max_tokens: int = 1024):
    model_name = selected_model['model_name']
    model_type = selected_model['model_type']

    model_config_cls = MODEL_CONFIG[model_type]['model_cls']
    model_config_pipeline = MODEL_CONFIG[model_type]['pipeline']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_config_pipeline.from_pretrained(model_name)  

    pipe = pipeline(model_config_cls,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens)
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

# === Streamlit Entry Function ===
def generate_answer(question: str, model_choice: str = "1") -> str:
    selected_model = MODELS.get(model_choice)
    llm = load_llm_model(selected_model)
    rag_prompt = create_rag_prompt(question, db_retriever=db, embedder=embedder, k_docs=10)
    return llm(rag_prompt)

