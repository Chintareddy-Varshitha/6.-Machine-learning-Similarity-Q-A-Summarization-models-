from flask import Flask, request, jsonify
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import string

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df = pd.read_excel('/Users/varshithachintareddy/Desktop/file1.xlsx')
required_columns = ['id', 'text', 'embedding']
if not set(required_columns).issubset(df.columns):
    raise ValueError("The DataFrame should have 'id', 'text', and 'embedding' columns.")
df['embedding'] = df['embedding'].apply(literal_eval)

@app.route('/similarity', methods=['GET'])
def get_similarity():
    try:
        input_text = request.args.get('inputtext')
        if input_text is None:
            raise ValueError("Please provide a valid input text using the 'inputtext' parameter.")
        
        input_embedding = model.encode(input_text, convert_to_tensor=True)
        input_embedding = input_embedding.flatten().tolist()

        similarity_scores = []
        for _, row in df.iterrows():
            row_vector = np.array(row['embedding']).flatten().tolist()
            if len(row_vector) == 0:
                raise ValueError("Empty embedding vector found in the DataFrame.")
            similarity_vector = cosine_similarity([input_embedding], [row_vector])[0]
            similarity_scores.append(similarity_vector[0])
        df['similarity_score'] = similarity_scores
        result_df = df[['id', 'embedding', 'similarity_score']] 
        result_html = result_df.to_html(index=False)
        result_html = f"User Input: {input_text}<br><br>{result_html}"
        return f"{result_html}"
    except ValueError as e:
        return jsonify({"error": str(e)}), 400 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5013)
