from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_system(queries_file, retrieved_docs_file):
    # Cargar consultas y documentos recuperados
    queries = pd.read_csv(queries_file)
    retrieved_docs = pd.read_csv(retrieved_docs_file)

    # Calcular métricas (ejemplo simplificado)
    precision = precision_score(queries['relevant'], retrieved_docs['relevant'], average='micro')
    recall = recall_score(queries['relevant'], retrieved_docs['relevant'], average='micro')
    f1 = f1_score(queries['relevant'], retrieved_docs['relevant'], average='micro')

    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    evaluate_system('data/queries_eval.csv', 'results/retrieved_docs/retrieved.csv')