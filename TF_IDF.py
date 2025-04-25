import json
import pandas as pd
import numpy as np
import os
import re
import time
import logging
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("legislation_retrieval.log"), logging.StreamHandler()]
)
logger = logging.getLogger("legislation_retrieval")

# ---------------------------
# Utility: Preprocessing
# ---------------------------
def preprocess_text(text):
    """Preprocess text for better retrieval performance."""
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove numbers (optional)
    text = re.sub(r'\d+', '', text)
    
    # Remove common stopwords (optional)
    stopwords = ['the', 'and', 'or', 'of', 'to', 'in', 'a', 'for', 'with', 'by', 'on', 'at']
    word_tokens = text.split()
    filtered_text = [word for word in word_tokens if word not in stopwords]
    text = ' '.join(filtered_text)
    
    # Stemming or lemmatization could be added here
    # This would require importing nltk or spacy libraries
    
    return text


# ---------------------------
# Step 1: Preprocessing & Saving with LSA
# ---------------------------
def preprocess_legislation_data(jsonl_path,
                               processed_path='preprocessed_legislation.pkl',
                               vectorizer_path='tfidf_vectorizer.pkl',
                               svd_path='svd_model.pkl',
                               lsa_matrix_path='lsa_matrix.npy',
                               n_components=100):
    """Preprocess legislation data and apply TF-IDF and LSA transformations."""
    try:
        logger.info("Preprocessing legislation data. This may take a while...")
        start_time = time.time()
        
        # Load JSONL and convert to DataFrame
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} documents.")
        
        # Preprocess text and create preview
        logger.info("Preprocessing text...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        df['text_preview'] = df['text'].apply(lambda x: x[:200] + '...' if len(x) > 200 else x)
        
        # Extract title from the first line or first sentence
        df['title'] = df['text'].apply(lambda x: x.split('\n')[0] if '\n' in x else (x.split('.')[0] if '.' in x else x[:50]))
        
        # Save processed DataFrame
        df[['id', 'year', 'processed_text', 'text_preview', 'title']].to_pickle(processed_path)
        logger.info(f"Processed DataFrame saved to {processed_path}")
        
        # TF-IDF Vectorization
        logger.info("Fitting TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.85, min_df=2, stop_words='english',
            use_idf=True, norm='l2', ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Apply LSA using TruncatedSVD for dimensionality reduction
        logger.info(f"Applying LSA to {n_components} components...")
        svd_model = TruncatedSVD(
            n_components=n_components, algorithm='randomized', 
            n_iter=7, random_state=42
        )
        lsa_matrix = svd_model.fit_transform(tfidf_matrix)
        logger.info(f"LSA matrix shape: {lsa_matrix.shape}")
        logger.info(f"Explained variance ratio sum: {svd_model.explained_variance_ratio_.sum():.4f}")
        
        # Save models and matrices
        import pickle
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(svd_path, 'wb') as f:
            pickle.dump(svd_model, f)
        np.save(lsa_matrix_path, lsa_matrix)
        logger.info("TF-IDF vectorizer, SVD model, and LSA matrix saved.")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds.")
        
        return df, tfidf_vectorizer, svd_model, lsa_matrix
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

# ---------------------------
# Step 2: Loading Preprocessed Data
# ---------------------------
def load_preprocessed_data(processed_path='preprocessed_legislation.pkl',
                          vectorizer_path='tfidf_vectorizer.pkl',
                          svd_path='svd_model.pkl',
                          lsa_matrix_path='lsa_matrix.npy'):
    """Load preprocessed data, vectorizer, SVD model, and LSA matrix."""
    try:
        required_files = [
            processed_path,
            vectorizer_path,
            svd_path,
            lsa_matrix_path
        ]
        
        if not all(os.path.exists(p) for p in required_files):
            logger.warning("Preprocessed files not found.")
            return None, None, None, None
            
        logger.info("Loading preprocessed data...")
        df = pd.read_pickle(processed_path)
        
        import pickle
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        with open(svd_path, 'rb') as f:
            svd_model = pickle.load(f)
            
        lsa_matrix = np.load(lsa_matrix_path)
        logger.info("Successfully loaded all preprocessed data.")
        
        return df, tfidf_vectorizer, svd_model, lsa_matrix
        
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {str(e)}")
        raise

# ---------------------------
# Step 3: Retrieval Function with LSA
# ---------------------------
def retrieve_relevant_legislation(query, df, tfidf_vectorizer, svd_model, lsa_matrix, top_n=3):
    """Retrieve relevant legislation based on the query."""
    try:
        if not query.strip():
            return []
            
        processed_query = preprocess_text(query)
        
        # Transform query using TF-IDF vectorizer
        query_vector = tfidf_vectorizer.transform([processed_query])
        
        # Apply LSA transformation to query vector
        query_lsa = svd_model.transform(query_vector)
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_lsa, lsa_matrix).flatten()
        
        # Get indices of top n most similar documents
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        # Filter out results with similarity below threshold
        min_similarity = 0.1  # Minimum similarity threshold
        results = []
        
        for idx in top_indices:
            if similarity_scores[idx] >= min_similarity:
                # Extract title from the preview text
                text_preview = df.iloc[idx]['text_preview']
                title = text_preview.split('.')[0] if '.' in text_preview else text_preview.split('\n')[0]
                
                results.append({
                    'id': df.iloc[idx]['id'],
                    'year': df.iloc[idx]['year'],
                    'similarity': similarity_scores[idx],
                    'text_preview': text_preview,
                    'title': title  # Add title to the dictionary
                })
                
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving legislation: {str(e)}")
        return []


# ---------------------------
# Step 4: Main Application
# ---------------------------
def main(df, tfidf_vectorizer, svd_model, lsa_matrix):
    """Run the system in interactive mode."""
    print("\n" + "="*50)
    print("UK Legislation Retrieval System")
    print("="*50)
    print("Enter a case description to find relevant UK legislation.")
    print("Commands:")
    print("  'exit' - Quit the program")
    print("  'test' - Run a quick test with sample queries")
    print("  'help' - Show this help message")
    print("="*50)
    
    while True:
        try:
            query = input("\nEnter case description: ")
            
            if query.lower() == 'exit':
                print("Exiting program. Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n" + "="*50)
                print("UK Legislation Retrieval System - Help")
                print("="*50)
                print("Enter a case description to find relevant UK legislation.")
                print("Commands:")
                print("  'exit' - Quit the program")
                print("  'test' - Run a quick test with sample queries")
                print("  'help' - Show this help message")
                print("="*50)
                continue
                
            if query.lower() == 'test':
                test_queries = ["tax law"]
                print("\n=== RUNNING TEST QUERIES ===")
                
                for test_query in test_queries:
                    print(f"\nTest query: '{test_query}'")
                    start_time = time.time()
                    results = retrieve_relevant_legislation(test_query, df, tfidf_vectorizer, svd_model, lsa_matrix)
                    elapsed_time = time.time() - start_time
                    
                    print(f"\nResults for: '{test_query}' (found in {elapsed_time:.3f} seconds)")
                    print("-" * 50)
                    
                    if not results:
                        print("No relevant legislation found.")
                    else:
                        for i, result in enumerate(results, 1):
                            try:
                                title = result.get('title', 'No title available')
                                
                                print(f"{i}. {title}")
                                print(f"   Year: {result['year']} | Law Code: {result['id']}")
                                print(f"   Relevance: {result['similarity']*100:.2f}%")
                                print(f"   Preview: {result['text_preview']}")
                                print("-" * 50)
                            except KeyError as e:
                                print(f"Error displaying result {i}: Missing field {e}")
                
                print("\n=== TEST COMPLETE ===")
                continue
            
            if not query.strip():
                print("Please enter a valid case description.")
                continue
            
            start_time = time.time()
            results = retrieve_relevant_legislation(query, df, tfidf_vectorizer, svd_model, lsa_matrix)
            elapsed_time = time.time() - start_time
            
            print(f"\nResults for: '{query}' (found in {elapsed_time:.3f} seconds)")
            print("-" * 50)
            
            if not results:
                print("No relevant legislation found.")
            else:
                for i, result in enumerate(results, 1):
                    try:
                        title = result.get('title', 'No title available')
                        
                        print(f"{i}. {title}")
                        print(f"   Year: {result['year']} | Law Code: {result['id']}")
                        print(f"   Relevance: {result['similarity']*100:.2f}%")
                        print(f"   Preview: {result['text_preview']}")
                        print("-" * 50)
                    except KeyError as e:
                        print(f"Error displaying result {i}: Missing field {e}")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            continue
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"An error occurred: {str(e)}")

# ---------------------------
# Step 5: Entry Point
# ---------------------------
if __name__ == "__main__":
    jsonl_path = os.path.join('uk_legislation/train.jsonl')
    processed_path = 'preprocessed_legislation.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    svd_path = 'svd_model.pkl'
    lsa_matrix_path = 'lsa_matrix.npy'

    try:
        # Load preprocessed data
        df, tfidf_vectorizer, svd_model, lsa_matrix = load_preprocessed_data(
            processed_path, vectorizer_path, svd_path, lsa_matrix_path
        )
        
        # If preprocessed files don't exist, create them
        if df is None:
            print("Preprocessed files not found. Running preprocessing...")
            df, tfidf_vectorizer, svd_model, lsa_matrix = preprocess_legislation_data(
                jsonl_path, processed_path, vectorizer_path, svd_path, lsa_matrix_path, n_components=100
            )
        
        # Start the main interactive system
        main(df, tfidf_vectorizer, svd_model, lsa_matrix)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"A fatal error occurred: {str(e)}")
