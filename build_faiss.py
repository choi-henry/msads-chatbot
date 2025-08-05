# build_faiss.py
from rag_pipeline import embed_chunk_docs, store_embed_in_db
import pandas as pd
import ast

df = pd.read_csv("Data/scraped_output_metadata.csv")
df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

from rag_pipeline import build_chunks_dataframe
df_chunks = build_chunks_dataframe(df)

all_chunks = df_chunks['chunk_text'].tolist()
all_metas = df_chunks['metadata'].tolist()

embedder, _ = embed_chunk_docs(all_chunks, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

store_embed_in_db(all_chunks, all_metas, embedder=embedder, persist_directory="faiss_index")
