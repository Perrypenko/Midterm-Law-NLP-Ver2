# Importing libraries
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Path to the JSONL file
jsonl_path = os.path.join('uk_legislation/train.jsonl')

# Function that loads JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Loading the JSONL file & converting to DataFrame
legislation_data = load_jsonl(jsonl_path)
legislation_df = pd.DataFrame(legislation_data)

# Extract text and metadata
legislation_ids = legislation_df['id'].tolist()
legislation_year = legislation_df['year'].tolist()
legislation_text = legislation_df['text'].tolist()

# Basic preprocessing function to test prototype
def preprocess_text(text):
    text = text.lower()
    return text

processed_texts = [preprocess_text(text) for text in legislation_text]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.85,       # Ignore terms that appear in more than 85% of documents
    min_df=2,          # Ignore terms that appear in fewer than 2 documents
    stop_words='english',
    use_idf=True,
    norm='l2',
    ngram_range=(1, 2) # Include both unigrams and bigrams
)

# Fit and transform the legislation texts
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}") 
print(f"Number of unique terms: {len(feature_names)}")

def retrieve_relevant_legislation(query, top_n=5):
    """
    Retrieve the most relevant legislation documents for a given query.
    
    Args:
        query (str): The user's case description
        top_n (int): Number of top results to return
        
    Returns:
        list: Top n relevant legislation documents with their similarity scores
    """
    # Preprocess the query
    processed_query = preprocess_text(query)
    
    # Transform the query using the same vectorizer
    query_vector = tfidf_vectorizer.transform([processed_query])
    
    # Calculate cosine similarity between query and all legislation documents
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top n most similar documents
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Prepare results
    results = []
    for idx in top_indices:
        results.append({
            'id': legislation_ids[idx],
            'year': legislation_year[idx],
            'similarity': similarity_scores[idx],
            'text_preview': legislation_text[idx][:200] + '...'  # Preview of the text
        })
    
    return results

def main():
    print("UK Legislation Retrieval System")
    print("-------------------------------")
    print("Enter a case description to find relevant UK legislation.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter case description: ")
        
        if query.lower() == 'exit':
            break
        
        if not query.strip():
            print("Please enter a valid case description.")
            continue
        
        results = retrieve_relevant_legislation(query)
        
        print("\nTop relevant legislation:")
        print("-------------------------")
        
        if not results:
            print("No relevant legislation found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['year']} (Similarity: {result['similarity']:.4f})")
                print(f"   ID: {result['id']}")
                print(f"   Preview: {result['text_preview']}")
                print()

if __name__ == "__main__":
    main()
