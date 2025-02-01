import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Descargar recursos de NLTK si no están instalados
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Limpia el texto eliminando caracteres no alfabéticos y espacios extra."""
    text = re.sub(r'\W+', ' ', text)  # Sustituye caracteres no alfabéticos por espacio
    text = text.lower().strip()  # Convierte a minúsculas y elimina espacios extra
    return text

def tokenize_text(text):
    """Tokeniza el texto y elimina stopwords en español."""
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('spanish')]
    return ' '.join(tokens)

def preprocess_corpus(input_file, output_file):
    """Preprocesa el archivo CSV, limpiando y tokenizando el campo 'Descripción'."""
    try:
        # Leer el archivo CSV con delimitador correcto y manejo de comillas
        df = pd.read_csv(input_file, encoding='utf-8', quotechar='"')

        # Asegurar que la columna 'Descripción' sea de tipo texto
        df['Descripción'] = df['Descripción'].astype(str).apply(clean_text)
        df['Descripción'] = df['Descripción'].apply(tokenize_text)

        # Guardar el nuevo CSV procesado
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Archivo procesado guardado en: {output_file}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'.")
    except pd.errors.ParserError as e:
        print(f"Error de parsing en el archivo CSV: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    preprocess_corpus('data/corpus.csv', 'data/corpus_preprocessed.csv')
