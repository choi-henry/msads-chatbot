# build_faiss.py

import os
import pandas as pd
import ast

from rag_pipeline import (
    embed_chunk_docs,
    store_embed_in_db,
    build_chunks_dataframe
)

def build_faiss_index():
    # 1. Robust path handling for cloud deployments
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Data", "scraped_output_metadata.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found at: {data_path}")

    # 2. Load and parse metadata
    df = pd.read_csv(data_path)
    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 3. Chunking
    df_chunks = build_chunks_dataframe(df)
    all_chunks = df_chunks['chunk_text'].tolist()
    all_metas = df_chunks['metadata'].tolist()

    # 4. Embedding
    embedder, _ = embed_chunk_docs(
        all_chunks,
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )

    # 5. Store to FAISS index
    store_embed_in_db(
        all_chunks,
        all_metas,
        embedder=embedder,
        persist_directory=os.path.join(base_dir, "faiss_index")
    )

    print("✅ FAISS index successfully built and saved.")

# 6. Allow script execution
if __name__ == "__main__":
    try:
        build_faiss_index()
    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}")
