#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[ ]:


import os
print("🔹 CONDA env:", os.environ.get("CONDA_DEFAULT_ENV"))


# In[ ]:


import os
print("🔹 CONDA env:", os.environ.get("CONDA_DEFAULT_ENV"))


# In[ ]:


# !pip install --quiet --upgrade datasets langchain langchain-community langchain-chroma sentence-transformers umap-learn matplotlib scikit-learn
# !pip install --upgrade PyMuPDF
# !pip install huggingface_hub


# In[ ]:


# import sys
# print(sys.executable)


# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity


# ## Load Data

# In[ ]:


df = pd.read_csv("../Data/scraped_output_metadata.csv")


# In[ ]:


df.info()


# In[ ]:


# df = df.rename(columns= {'scraped_Content': 'scraped_content'})


# In[ ]:


# df_filtered = df.drop(index = [4, 5, 14, 15, 24, 25])


# In[ ]:


# df_filtered_reset_index = df_filtered.reset_index()
# df_filtered_reset_index = df_filtered_reset_index.drop(columns = 'index')
# df_filtered_reset_index.info()


# In[ ]:


# df_filtered_reset_index.sample(2)


# In[ ]:


# df = df_filtered_reset_index.copy()


# In[ ]:


pd.set_option("display.max_colwidth", 500)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


import ast

df['metadata'] = df['metadata'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)


# In[ ]:


df.info()


# ## Document Indexing

# ### Chunking

# In[ ]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Iterable, List

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def build_chunks_dataframe(
    df: pd.DataFrame,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> pd.DataFrame:

    records = []

    for source_idx, row in df.iterrows():
        seg_text = row['text']
        seg_metadata = row['metadata']

        seg_chunks = chunk_text(seg_text, chunk_size, chunk_overlap)

        for chunk_idx, chunk in enumerate(seg_chunks):
            meta = seg_metadata.copy()
            meta['source_idx'] = source_idx
            meta['chunk_idx'] = chunk_idx

            records.append({'chunk_text': chunk,
                            'metadata': meta})

    return pd.DataFrame(records)


# In[ ]:


df_chunks = build_chunks_dataframe(df)


# In[ ]:


df_chunks


# In[ ]:


# df['chunk_text'] = df['scraped_content'].apply(lambda x: chunk_text(x, chunk_size = 800, chunk_overlap = 80))


# In[ ]:


# df_chunks['chunk_count'] = df_chunks['chunk_text'].apply(len)


# In[ ]:


# from itertools import chain

# all_chunks = list(chain.from_iterable(df_chunks['chunk_text']))

# total_chunks = len(all_chunks)
# total_doc = len(df_chunks)
# print(f"Total Chunk Texts: {total_chunks}")
# print(f"Average Chunk per Document: {(total_chunks / total_doc):.2f}")


# In[ ]:


all_chunks = df_chunks['chunk_text'].tolist()
all_metas = df_chunks['metadata'].tolist()


# In[ ]:


len(all_chunks)


# In[ ]:


all_chunks[:5]


# In[ ]:


all_metas[:5]


# ### Embedding

# In[ ]:


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Setting device to '{device}'")


# In[ ]:


from typing import List
# Use a pre-trained Sentence-BERT model to convert each chunk into a semantic vector
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedder = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

def embed_chunk_docs(chunk_docs: List[str],
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     normalize: bool = True):
    """
    Embed chunks of documents based on a given model name.
    """

    embedder = HuggingFaceEmbeddings(
    model_name= model_name,
    encode_kwargs={"normalize_embeddings": normalize})

    embeddings = embedder.embed_documents(chunk_docs)

    return embedder, embeddings


# In[ ]:


# embedder, embeddings = embed_chunk_docs(all_chunks, "intfloat/e5-large-v2", True)
# embedder, embeddings = embed_chunk_docs(all_chunks, "sentence-transformers/all-MiniLM-L6-v2", True)
embedder, embeddings = embed_chunk_docs(all_chunks, "sentence-transformers/paraphrase-MiniLM-L6-v2", True)


# In[ ]:


len(embeddings)


# In[ ]:


len(embeddings[0])


# ### Store Embeddings in Chroma DB

# In[ ]:


# !rm -rf ./chroma_scifact


# In[ ]:


from langchain_chroma import Chroma
# from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import time
from pathlib import Path
from typing import Optional, Union


def store_embed_in_db(texts: List[str],
    metas: List[dict],
    embedder: Optional[Embeddings] = None,
    persist_directory: Optional[Union[str, Path]] = None
) -> Chroma:
    """
    Store text chunks (and their embeddings) in a Chroma vector database.

    You must provide *either* `embeddings` (precomputed) *or* an `embedder`
    (model that implements .embed_documents).

    Args:
        texts: List of text chunks to index.
        embedder: Optional Embeddings object; used to compute embeddings on the fly.
        persist_directory: Directory where the Chroma DB will be saved.
            If None, a timestamped folder will be created.

    Returns:
        A Chroma vector store instance with your texts indexed.
    """

    if embedder is None:
        raise ValueError("You must supply an `embedder`")

    if persist_directory is None:
        ts = int(time.time())
        persist_directory = Path(f"./chroma_scifact_run_{ts}")
    persist_directory = Path(persist_directory)

    all_metas = df_chunks['metadata'].tolist()


    db = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas = metas,
        persist_directory=str(persist_directory)
    )

    print("Chunks successfully stored in Chroma vector DB.")

    return db


# In[ ]:


db = store_embed_in_db(all_chunks, all_metas, embedder = embedder)


# In[ ]:


col = db._collection  # the underlying chromadb Collection
data = col.get()


# In[ ]:


len(data['documents'])


# In[ ]:


meta_section_set = set()

for meta in all_metas:
    meta_section_set.add(meta.get('section', ''))


# In[ ]:


meta_section_set


# In[ ]:


meta_super_section_set = set()

for meta in all_metas:
    meta_super_section_set.add(meta.get('super_section', ''))


# In[ ]:


meta_super_section_set


# In[ ]:


df_chunks[df_chunks['metadata'].apply(lambda x: x.get('section', '') == 'Noncredit Courses')]


# In[ ]:


# all_metas.get('section', '')


# ## Query-Response Pipeline (RAG)

# In[ ]:


# Improved retrieval function with query expansion and better ranking
from langchain.vectorstores.base import VectorStoreRetriever
import re
from typing import List, Tuple, Dict, Any

def expand_query(query: str) -> List[str]:
    """Expand query with related terms for better retrieval"""
    expansions = {
        "core courses": ["required courses", "core curriculum", "fundamental courses",
                        "mandatory courses", "core classes", "required classes"],
        "admission requirements": ["application requirements", "prerequisites",
                                 "eligibility", "admission criteria"],
        "capstone": ["final project", "thesis", "culminating project", "capstone project"],
        "curriculum": ["courses", "program structure", "degree requirements"],
        "machine learning": ["ML", "artificial intelligence", "AI", "deep learning"],
        "data science": ["analytics", "data analytics", "data mining", "statistics"]
    }

    expanded_queries = [query]
    query_lower = query.lower()

    for key, synonyms in expansions.items():
        if key in query_lower:
            for synonym in synonyms:
                expanded_queries.append(query.replace(key, synonym, 1))

    return expanded_queries

def detect_metadata_filter(query: str) -> Optional[Dict[str, Any]]:
    """
    Look for trigger phrases in the query and return an appropriate
    metadata filter for your vector store.
    """
    q = query.lower()
    mapping = {
        "core course":     {"section": {"$eq": "Core Courses"}},
        "elective":        {"section": {"$eq": "Sample Elective Courses"}},
        "foundation":      {"section": {"$eq": "Noncredit Courses"}},
        "capstone sponsor":{"section": {"$eq": "MS-ADS Capstone Sponsor Guide 2025"}},
        "application":     {"super_section": {"$eq": "Master’s in Applied Data Science Application Requirements"}},
        "admission":       {"super_section": {"$eq": "Master’s in Applied Data Science Application Requirements"}}

    }

    for trigger, filt in mapping.items():
        if trigger in q:
            return filt
    return None


def retrive_top_k_docs_improved(selected_query: str,
                               db_retriever: VectorStoreRetriever,
                               embedder: Embeddings,
                               k_docs: int = 10,  # Increased from 5
                               lambda_mult: float = 0.3,  # Lower for more similarity focus
                               query_prefix: str = "query: ",
                               doc_prefix: str = "passage: ",
                               use_query_expansion: bool = True):
    """
    Improved retrieval with query expansion and better re-ranking
    """

    if embedder is None:
        raise ValueError("You must supply an `embedder`.")

    # Step 1: Query expansion
    if use_query_expansion:
        expanded_queries = expand_query(selected_query)
    else:
        expanded_queries = [selected_query]

    # Step 2: Auto‐detect a metadata filter
    metadata_filter = detect_metadata_filter(selected_query)

    # Step 3: Retrieve documents for each expanded query with metadata filter
    all_retrieved_docs = []

    for exp_query in expanded_queries:
        formatted_query = f"{query_prefix}{exp_query}"

        # Define retriever method
        retriever = db_retriever.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_docs,
                           "lambda_mult": lambda_mult,
                          **({"filter": metadata_filter} if metadata_filter else {})}
        )

        # Get relevant documents
        results = retriever.get_relevant_documents(formatted_query)

        # Store documents with their expanded query context
        for doc in results:
            all_retrieved_docs.append((doc.page_content, exp_query))

    # Step 3: Remove duplicates while preserving order
    seen_docs = set()
    unique_docs = []
    for doc_content, exp_query in all_retrieved_docs:
        if doc_content not in seen_docs:
            seen_docs.add(doc_content)
            unique_docs.append((doc_content, exp_query))

    # Step 4: Re-rank all unique documents against original query
    original_formatted_query = f"{query_prefix}{selected_query}"
    query_embed = embedder.embed_query(original_formatted_query)

    # Format documents for embedding
    formatted_retrieved_docs = [f"{doc_prefix}{doc}" for doc, _ in unique_docs]

    # Embed documents
    doc_embeds = embedder.embed_documents(formatted_retrieved_docs)

    # Compute similarity scores
    similarity_scores = cosine_similarity([query_embed], doc_embeds)[0]

    # Step 5: Combine documents with scores and rank
    ranked_docs = []
    for i, (formatted_doc, score) in enumerate(zip(formatted_retrieved_docs, similarity_scores)):
        original_doc, exp_query = unique_docs[i]
        ranked_docs.append((formatted_doc, score, exp_query))

    # Sort by similarity score (descending)
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    # Step 6: Apply keyword boost for exact matches
    boosted_docs = []
    query_keywords = set(re.findall(r'\b\w+\b', selected_query.lower()))

    for doc, score, exp_query in ranked_docs:
        doc_keywords = set(re.findall(r'\b\w+\b', doc.lower()))
        keyword_overlap = len(query_keywords.intersection(doc_keywords))

        # Boost score based on keyword overlap
        boosted_score = score + (keyword_overlap * 0.05)  # Small boost for keyword matches
        boosted_docs.append((doc, boosted_score, exp_query))

    # Final sort by boosted scores
    boosted_docs.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("Query:", selected_query)
    print(f"\nExpanded to {len(expanded_queries)} queries")
    print(f"Retrieved {len(unique_docs)} unique documents")
    print(f"\nTop {k_docs} Retrieved Chunks with Scores:\n")

    final_results = []
    for i, (doc, score, exp_query) in enumerate(boosted_docs[:k_docs]):
        print(f"Chunk {i+1} (Score: {score:.4f}, From: '{exp_query}'):")
        print(f"{doc}")
        print(f"{'-'*80}")
        final_results.append((doc, score))

    return boosted_docs[:k_docs]


# In[ ]:


# Compare original vs improved retrieval with question selection
QUERIES = {"1": "What are the core courses in the MS in Applied Data Science program?",
           "2": "What are the admission requirements for the MS in Applied Data Science program?",
           "3": "Can you provide information about the capstone project?"}

print("Available queries:")
for key, value in QUERIES.items():
    print(f"{key}: {value}")

query_choice = input("Enter number of your query choice: ")
selected_query = QUERIES.get(query_choice)

if selected_query:
    print(f"\nSelected query: {selected_query}")

    print("\n" + "=" * 50)
    print("IMPROVED RETRIEVAL METHOD")
    print("=" * 50)

    result_improved = retrive_top_k_docs_improved(selected_query,
                       db_retriever = db,
                       embedder = embedder,
                       k_docs = 10,
                       lambda_mult = 0.3,
                       query_prefix = 'query: ',
                       doc_prefix = 'passage: ',
                       use_query_expansion = True)
else:
    print("Invalid choice. Please select 1, 2, or 3.")


# In[ ]:


from langchain.vectorstores.base import VectorStoreRetriever

def create_rag_prompt(selected_query: str,
                       db_retriever: VectorStoreRetriever,
                       embedder: Embeddings,
                       k_docs: int = 10,
                       lambda_mult: float = 0.3,
                       query_prefix: str = "query: ",
                       doc_prefix: str = "passage: ",
                       use_query_expansion: bool = True):
    """
    Retrieve the top-k most relevant chunks for a given query using
    Maximal Marginal Relevance, then re-rank by cosine similarity.

    Args:
        query: Raw query string.
        db_retriever: The `db.as_retriever(...)` object configured with MMR.
        embedder: An Embeddings instance (implements embed_query & embed_documents).
        k: How many chunks to return.
        lambda_mult: MMR diversity parameter (0→pure similarity, 1→pure diversity).
        query_prefix: Text to prepend before embedding the query.
        doc_prefix: Text to prepend before embedding each chunk.

    Returns:
        A list of (chunk_text, score) sorted descending by score.
    """

    if embedder is None:
        raise ValueError("You must supply an `embedder`.")

    # Step 1: Query expansion
    if use_query_expansion:
        expanded_queries = expand_query(selected_query)
    else:
        expanded_queries = [selected_query]

    # Step 2: Auto‐detect a metadata filter
    metadata_filter = detect_metadata_filter(selected_query)

    # Step 3: Retrieve documents for each expanded query with metadata filter
    all_retrieved_docs = []

    for exp_query in expanded_queries:
        formatted_query = f"{query_prefix}{exp_query}"

        # Define retriever method
        retriever = db_retriever.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_docs,
                           "lambda_mult": lambda_mult,
                          **({"filter": metadata_filter} if metadata_filter else {})}
        )

        # Get relevant documents
        results = retriever.get_relevant_documents(formatted_query)

        # Store documents with their expanded query context
        for doc in results:
            all_retrieved_docs.append((doc.page_content, exp_query))

    # Step 4: Remove duplicates while preserving order
    seen_docs = set()
    unique_docs = []
    for doc_content, exp_query in all_retrieved_docs:
        if doc_content not in seen_docs:
            seen_docs.add(doc_content)
            unique_docs.append((doc_content, exp_query))

    # Step 5: Re-rank all unique documents against original query
    original_formatted_query = f"{query_prefix}{selected_query}"
    query_embed = embedder.embed_query(original_formatted_query)

    # Format documents for embedding
    formatted_retrieved_docs = [f"{doc_prefix}{doc}" for doc, _ in unique_docs]

    # Embed documents
    doc_embeds = embedder.embed_documents(formatted_retrieved_docs)

    # Compute similarity scores
    similarity_scores = cosine_similarity([query_embed], doc_embeds)[0]

    # Step 6: Combine documents with scores and rank
    ranked_docs = []
    for i, (formatted_doc, score) in enumerate(zip(formatted_retrieved_docs, similarity_scores)):
        original_doc, exp_query = unique_docs[i]
        ranked_docs.append((formatted_doc, score, exp_query))

    # Sort by similarity score (descending)
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    # Step 7: Apply keyword boost for exact matches
    boosted_docs = []
    query_keywords = set(re.findall(r'\b\w+\b', selected_query.lower()))

    for doc, score, exp_query in ranked_docs:
        doc_keywords = set(re.findall(r'\b\w+\b', doc.lower()))
        keyword_overlap = len(query_keywords.intersection(doc_keywords))

        # Boost score based on keyword overlap
        boosted_score = score + (keyword_overlap * 0.05)  # Small boost for keyword matches
        boosted_docs.append((doc, boosted_score, exp_query))

    # Final sort by boosted scores
    boosted_docs.sort(key=lambda x: x[1], reverse=True)

    # Combine the retrieved context into a single block of text
    context = "\n".join([doc for doc, score, exp_query in boosted_docs])

    # Compose the RAG prompt by combining the context and the user's question
    rag_prompt = f"""
    You are an educational assistant.
    Your task is to extract information related to Master of Science in Applied Data Science program, such as admission requirement, curriculum, faculties, career outcomes, and etc.
    Use the following context to answer the question.

    [CONTEXT]
    {context}

    [QUESTION]
    {formatted_query}

    Summarize your answers that are easy to follow. Feel free to use a combination of paragraph and bullet points for your answers
    """

    # Display the final composed prompt that will be passed to the LLM
    print("=== RAG Augmented Prompt ===")
    print(rag_prompt)

    return rag_prompt


# In[ ]:


# Define a new query
QUERIES = {"1": "What are the core courses in the MS in Applied Data Science program?",
 "2" : "What are the admission requirements for the MS in Applied Data Science program?",
 "3" : "Can you provide information about the capstone project?"
}

query_choice = input("Enter number of your query choice")
selected_query = QUERIES.get(query_choice)

rag_prompt = create_rag_prompt(selected_query,
                   db_retriever = db,
                   embedder = embedder,
                   k_docs = 10,
                   lambda_mult = 0.3,
                   query_prefix = 'query: ',
                   doc_prefix = 'passage: ',
                   use_query_expansion = True)


# In[ ]:


# !pip install huggingface_hub
# huggingface-cli login


# In[ ]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

MODELS = {"1": {"model_name":"google/flan-t5-base", "model_type": "encoder-decoder"},
          "2": {"model_name":"mistralai/Mistral-7B-Instruct-v0.3", "model_type": "decoder only"},
          "3": {"model_name":"meta-llama/Llama-2-7b-hf", "model_type": "decoder only"}}

MODEL_CONFIG = {"encoder-decoder": {"model_cls": "text2text-generation",
                                    "pipeline": AutoModelForSeq2SeqLM},
                "decoder only": {"model_cls": "text-generation",
                                 "pipeline": AutoModelForCausalLM}}


# In[ ]:


def load_llm_model(selected_model: dict,
                   max_tokens: int = 1024):
    model_name = selected_model['model_name']
    model_type = selected_model['model_type']

    model_config_cls = MODEL_CONFIG[model_type]['model_cls']
    model_config_pipeline = MODEL_CONFIG[model_type]['pipeline']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_config_pipeline.from_pretrained(model_name,
                                            device_map="auto",
                                            torch_dtype="auto")

    pipe = pipeline(model_config_cls,
                    model = model,
                    tokenizer = tokenizer,
                    max_new_tokens = max_tokens)
    llm = HuggingFacePipeline(pipeline = pipe)

    return llm


# In[ ]:


def main():
    choice = input("Enter number of your LLM model choice")
    selected_model = MODELS.get(choice)

    llm = load_llm_model(selected_model)

    query_choice = input("Enter number of your query choice: ")
    selected_query = QUERIES.get(query_choice)

    rag_prompt = create_rag_prompt(selected_query,
                   db_retriever = db,
                   embedder = embedder,
                   k_docs = 10,
                   lambda_mult = 0.3,
                   query_prefix = 'query: ',
                   doc_prefix = 'passage: ',
                   use_query_expansion = True
                   )

    response = llm(rag_prompt)
    print("\n=== Generated Answer ===")
    print(response)


# ## Question 1: What are the core courses in the MS in Applied Data Science program?

# In[ ]:


import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# In[ ]:


if __name__ == "__main__":
    main()


# ## Question 2: What are the admission requirements for the MS in Applied Data Science program?

# In[ ]:


if __name__ == "__main__":
    main()


# ## Question 3: Can you provide information about the capstone project??

# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


def generate_answer(question: str, model_choice: str = "1") -> str:
    selected_model = MODELS.get(model_choice)
    llm = load_llm_model(selected_model)

    rag_prompt = create_rag_prompt(question,
                   db_retriever = db,
                   embedder = embedder,
                   k_docs = 10,
                   lambda_mult = 0.3,
                   query_prefix = 'query: ',
                   doc_prefix = 'passage: ',
                   use_query_expansion = True
                   )

    response = llm(rag_prompt)
    return response




