from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Cargar embeddings y corpus
embeddings = np.load('data/embeddings/corpus_embeddings.npy')
df = pd.read_csv('data/corpus_preprocessed.csv')

# Crear índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Cargar modelo de recuperación
retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Cargar el modelo GPT-2 de Hugging Face (puedes cambiar a otro modelo como GPT-Neo si lo prefieres)
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.json
    query = data.get("query")

    # Recuperar documentos relevantes
    query_embedding = retrieval_model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    retrieved_docs = df.iloc[indices[0]]

    # Generar respuesta con GPT-2 de Hugging Face
    context = " ".join(retrieved_docs['Descripción'].tolist())
    input_text = f"Consulta: {query}\nContexto: {context}\nRespuesta:"

    # Tokenizar y generar texto
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=100)  # Limitar a tokens nuevos

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Devolver la respuesta y los documentos relevantes
    return jsonify({
        "response": generated_text,
        "documents": retrieved_docs.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)
