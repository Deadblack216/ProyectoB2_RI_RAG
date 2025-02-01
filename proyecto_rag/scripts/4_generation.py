from transformers import pipeline

def generate_response(query, context):
    # Cargar modelo de generación
    generator = pipeline('text-generation', model='gpt-3.5-turbo')

    # Generar respuesta
    prompt = f"Consulta: {query}\nContexto: {context}\nRespuesta:"
    response = generator(prompt, max_length=200)
    return response[0]['generated_text']

if __name__ == "__main__":
    query = "¿Cuál es el plan de salud del candidato X?"
    context = " ".join(retrieved_docs['Descripción'].tolist())  # Usar documentos recuperados
    response = generate_response(query, context)
    print(response)