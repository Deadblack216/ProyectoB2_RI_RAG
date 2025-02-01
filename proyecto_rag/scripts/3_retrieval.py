import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def retrieve_documents(query, embeddings_file, corpus_file, k=5):
    # Cargar embeddings y corpus
    embeddings = np.load(embeddings_file)
    df = pd.read_csv(corpus_file)

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Generar embedding de la consulta
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # Buscar documentos relevantes
    distances, indices = index.search(query_embedding, k)
    return df.iloc[indices[0]]

if __name__ == "__main__":
    query = "¿Cuál es el plan de salud del candidato X?"
    retrieved_docs = retrieve_documents(
        query,
        'models/embeddings/corpus_embeddings.npy',
        'data/corpus_preprocessed.csv'
    )
    print(retrieved_docs)