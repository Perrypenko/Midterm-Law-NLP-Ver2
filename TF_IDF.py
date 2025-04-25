import json
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ---------------------------
# Utility: Preprocessing
# ---------------------------
def preprocess_text(text):
    return text.lower()

# ---------------------------
# Step 1: Preprocessing & Saving with LSA
# ---------------------------
def preprocess_legislation_data(jsonl_path,
                               processed_path='preprocessed_legislation.pkl',
                               vectorizer_path='tfidf_vectorizer.pkl',
                               svd_path='svd_model.pkl',
                               lsa_matrix_path='lsa_matrix.npz',
                               n_components=100):
    print("Preprocessing large legislation file. This may take a while...")

    # Load JSONL and convert to DataFrame
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} documents.")

    # Preprocess text and create preview
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['text_preview'] = df['text'].apply(lambda x: x[:200] + '...')

    # Save processed DataFrame
    df[['id', 'year', 'processed_text', 'text_preview']].to_pickle(processed_path)
    print(f"Processed DataFrame saved to {processed_path}")

    # TF-IDF Vectorization
    print("Fitting TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.85, min_df=2, stop_words='english',
        use_idf=True, norm='l2', ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Apply LSA using TruncatedSVD for dimensionality reduction
    print(f"Applying LSA (dimensionality reduction) to {n_components} components...")
    svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=7, random_state=42)
    lsa_matrix = svd_model.fit_transform(tfidf_matrix)
    print(f"LSA matrix shape: {lsa_matrix.shape}")
    print(f"Explained variance ratio sum: {svd_model.explained_variance_ratio_.sum():.4f}")

    # Save vectorizer, SVD model, and LSA matrix
    import pickle
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(svd_path, 'wb') as f:
        pickle.dump(svd_model, f)
    np.save(lsa_matrix_path, lsa_matrix)
    print("TF-IDF vectorizer, SVD model, and LSA matrix saved.")
    
    return df, tfidf_vectorizer, svd_model, lsa_matrix

# ---------------------------
# Step 2: Loading Preprocessed Data
# ---------------------------
def load_preprocessed_data(processed_path='preprocessed_legislation.pkl',
                          vectorizer_path='tfidf_vectorizer.pkl',
                          svd_path='svd_model.pkl',
                          lsa_matrix_path='lsa_matrix.npy'):
    # Check for existence, else preprocess
    if not all(os.path.exists(p) for p in [processed_path, vectorizer_path, svd_path, lsa_matrix_path]):
        raise FileNotFoundError("Preprocessed files not found. Please run preprocessing first.")

    print("Loading preprocessed DataFrame...")
    df = pd.read_pickle(processed_path)

    print("Loading TF-IDF vectorizer...")
    import pickle
    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    print("Loading SVD model...")
    with open(svd_path, 'rb') as f:
        svd_model = pickle.load(f)

    print("Loading LSA matrix...")
    lsa_matrix = np.load(lsa_matrix_path)

    return df, tfidf_vectorizer, svd_model, lsa_matrix

# ---------------------------
# Step 3: Retrieval Function with LSA
# ---------------------------
def retrieve_relevant_legislation(query, df, tfidf_vectorizer, svd_model, lsa_matrix, top_n=5):
    processed_query = preprocess_text(query)
    
    # Transform query using TF-IDF vectorizer
    query_vector = tfidf_vectorizer.transform([processed_query])
    
    # Apply LSA transformation to query vector
    query_lsa = svd_model.transform(query_vector)
    
    # Calculate cosine similarity between query and all legislation documents in LSA space
    similarity_scores = cosine_similarity(query_lsa, lsa_matrix).flatten()
    
    # Get indices of top n most similar documents
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'id': df.iloc[idx]['id'],
            'year': df.iloc[idx]['year'],
            'similarity': similarity_scores[idx],
            'text_preview': df.iloc[idx]['text_preview']
        })
    return results

# ---------------------------
# Step 4: Main Application
# ---------------------------
def main(df, tfidf_vectorizer, svd_model, lsa_matrix):
    print("UK Legislation Retrieval System")
    print("-------------------------------")
    print("Enter a case description to find relevant UK legislation.")
    print("Type 'exit' to quit.")
    print("Type 'test' to run a quick test with a sample query.")
    
    while True:
        query = input("\nEnter case description: ")
        
        if query.lower() == 'exit':
            break
            
        if query.lower() == 'test':
            test_queries = ["tax law"]
            print("\n=== RUNNING TEST QUERIES ===")
            for test_query in test_queries:
                print(f"\nTest query: '{test_query}'")
                results = retrieve_relevant_legislation(test_query, df, tfidf_vectorizer, svd_model, lsa_matrix)
                print("Top relevant legislation:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['year']} (Similarity: {result['similarity']:.4f})")
                    print(f"   ID: {result['id']}")
                    print(f"   Preview: {result['text_preview']}")
            print("\n=== TEST COMPLETE ===")
            continue
        
        if not query.strip():
            print("Please enter a valid case description.")
            continue
        
        results = retrieve_relevant_legislation(query, df, tfidf_vectorizer, svd_model, lsa_matrix)
        
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
    jsonl_path = os.path.join('uk_legislation/train.jsonl')
    processed_path = 'preprocessed_legislation.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    svd_path = 'svd_model.pkl'
    lsa_matrix_path = 'lsa_matrix.npy'

    # If preprocessed files don't exist, create them
    if not all(os.path.exists(p) for p in [processed_path, vectorizer_path, svd_path, lsa_matrix_path]):
        print("Preprocessed files not found. Running preprocessing...")
        df, tfidf_vectorizer, svd_model, lsa_matrix = preprocess_legislation_data(
            jsonl_path, processed_path, vectorizer_path, svd_path, lsa_matrix_path, n_components=100
        )
    else:
        # Load preprocessed data
        df, tfidf_vectorizer, svd_model, lsa_matrix = load_preprocessed_data(
            processed_path, vectorizer_path, svd_path, lsa_matrix_path
        )
    
    # Test section to quickly check if the code works
    print("\n=== QUICK TEST ===")
    print("Running a sample query to test functionality...")
    sample_query = "tax law"
    results = retrieve_relevant_legislation(sample_query, df, tfidf_vectorizer, svd_model, lsa_matrix)
    
    print(f"\nResults for test query: '{sample_query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['year']} (Similarity: {result['similarity']:.4f})")
        print(f"   ID: {result['id']}")
        print(f"   Preview: {result['text_preview']}")
    
    print("\n=== TEST COMPLETE ===")
    print("Now starting the interactive system...")
    print()
    
    # Start the main interactive system
    main(df, tfidf_vectorizer, svd_model, lsa_matrix)
