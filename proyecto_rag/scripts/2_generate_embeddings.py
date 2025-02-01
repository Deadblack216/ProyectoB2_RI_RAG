import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def generate_embeddings(input_file, output_file):
    df = pd.read_csv(input_file)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['Descripción'].tolist())
    np.save(output_file, embeddings)

if __name__ == "__main__":
    generate_embeddings('data/corpus_preprocessed.csv', 'data/embeddings/corpus_embeddings.npy')