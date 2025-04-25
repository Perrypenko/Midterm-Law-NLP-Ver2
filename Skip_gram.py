import json
import numpy as np
from collections import Counter
import random
from sklearn.metrics.pairwise import cosine_similarity
import gc  # For garbage collection

class OptimizedSkipGram:
    def __init__(self, vector_size=100, window_size=2, learning_rate=0.025, epochs=5, min_count=5, max_vocab_size=50000):
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        self.W = None
        self.W_prime = None
        
    def build_vocab(self, word_counts=None):
        if word_counts:
            self.word_counts = word_counts
        
        # Filter by frequency and limit vocabulary size
        vocab = [word for word, count in self.word_counts.most_common(self.max_vocab_size) 
                if count >= self.min_count]
        
        print(f"Vocabulary size: {len(vocab)} words")
        
        # Create word mappings
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        # Initialize weights
        self.W = np.random.uniform(-0.1/self.vector_size, 0.1/self.vector_size, 
                                  (self.vocab_size, self.vector_size))
        self.W_prime = np.random.uniform(-0.1/self.vector_size, 0.1/self.vector_size, 
                                        (self.vector_size, self.vocab_size))
        
        # Clear memory
        del vocab
        gc.collect()
    
    def generate_training_data(self, corpus, subsample_threshold=1e-5):
        training_data = []
        total_words = sum(self.word_counts.values())
        
        for sentence in corpus:
            word_indices = []
            
            # Convert sentence to indices and apply subsampling
            for word in sentence:
                if word in self.word_to_idx:
                    # Subsampling frequent words
                    word_freq = self.word_counts[word] / total_words
                    prob = max(0, 1 - np.sqrt(subsample_threshold / word_freq))
                    
                    if random.random() > prob:
                        word_indices.append(self.word_to_idx[word])
            
            # Generate skip-gram pairs
            for i, center_idx in enumerate(word_indices):
                context_indices = list(range(max(0, i - self.window_size), i)) + \
                                 list(range(i + 1, min(len(word_indices), i + self.window_size + 1)))
                
                for context_idx in context_indices:
                    if 0 <= context_idx < len(word_indices):
                        training_data.append((center_idx, word_indices[context_idx]))
        
        print(f"Generated {len(training_data)} training pairs")
        return training_data
    
    def train(self, training_data, batch_size=2048, negative_samples=5):
        for epoch in range(self.epochs):
            loss = 0
            random.shuffle(training_data)
            
            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_loss = 0
                
                for center_idx, context_idx in batch:
                    # Forward pass for positive sample
                    h = self.W[center_idx]
                    u_positive = np.dot(h, self.W_prime[:, context_idx])
                    sigmoid_positive = 1 / (1 + np.exp(-u_positive))
                    
                    # Update for positive sample
                    gradient = self.learning_rate * (sigmoid_positive - 1)
                    self.W[center_idx] -= gradient * self.W_prime[:, context_idx]
                    self.W_prime[:, context_idx] -= gradient * h
                    
                    # Negative sampling
                    for _ in range(negative_samples):
                        neg_idx = random.randint(0, self.vocab_size - 1)
                        while neg_idx == context_idx:
                            neg_idx = random.randint(0, self.vocab_size - 1)
                        
                        u_negative = np.dot(h, self.W_prime[:, neg_idx])
                        sigmoid_negative = 1 / (1 + np.exp(-u_negative))
                        
                        # Update for negative sample
                        gradient = self.learning_rate * sigmoid_negative
                        self.W[center_idx] -= gradient * self.W_prime[:, neg_idx]
                        self.W_prime[:, neg_idx] -= gradient * h
                    
                    batch_loss -= np.log(sigmoid_positive)
                
                loss += batch_loss
                
                if i % (10 * batch_size) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size}, Progress: {i/len(training_data)*100:.1f}%")
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss/len(training_data)}")
    
    def get_word_vector(self, word):
        if word in self.word_to_idx:
            return self.W[self.word_to_idx[word]]
        return None
    
    def find_similar_words(self, word, top_n=10):
        if word not in self.word_to_idx:
            print(f"'{word}' not in vocabulary")
            return []
        
        word_vec = self.get_word_vector(word)
        word_idx = self.word_to_idx[word]
        
        # Compute similarities in batches to save memory
        similarities = []
        batch_size = 1000
        
        for i in range(0, self.vocab_size, batch_size):
            batch_indices = list(range(i, min(i + batch_size, self.vocab_size)))
            batch_vectors = self.W[batch_indices]
            
            # Compute cosine similarity
            sims = cosine_similarity([word_vec], batch_vectors)[0]
            
            for j, sim in enumerate(sims):
                idx = i + j
                if idx != word_idx:
                    similarities.append((self.idx_to_word[idx], sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def load_jsonl_sample(file_path, max_docs=5000, max_tokens_per_doc=1000):
    """Load a sample of the JSONL file with limits on document and token counts"""
    corpus = []
    word_counts = Counter()
    doc_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if doc_count >= max_docs:
                break
                
            data = json.loads(line)
            if 'text' in data:
                # Simple tokenization and limit tokens per document
                tokens = data['text'].lower().split()[:max_tokens_per_doc]
                corpus.append(tokens)
                
                # Update word counts
                for word in tokens:
                    word_counts[word] += 1
                
                doc_count += 1
    
    print(f"Loaded {len(corpus)} documents with {sum(len(doc) for doc in corpus)} tokens")
    return corpus, word_counts

def main():
    # File path to your JSONL file
    jsonl_file = "uk_legislation/train.jsonl"
    
    # Load a sample of the corpus
    print("Loading corpus sample...")
    corpus, word_counts = load_jsonl_sample(jsonl_file, max_docs=5000)
    
    # Initialize and train the optimized Skip-Gram model
    print("Training optimized Skip-Gram model...")
    model = OptimizedSkipGram(vector_size=100, window_size=2, learning_rate=0.025, 
                             epochs=3, min_count=5, max_vocab_size=20000)
    model.build_vocab(word_counts)
    
    training_data = model.generate_training_data(corpus)
    model.train(training_data, batch_size=2048, negative_samples=5)
    
    # Interactive query loop
    while True:
        query = input("\nEnter a legal term (or 'exit' to quit): ").strip().lower()
        if query == 'exit':
            break
        
        similar_terms = model.find_similar_words(query)
        if similar_terms:
            print(f"\nTerms similar to '{query}':")
            for term, score in similar_terms:
                print(f"- {term} (similarity: {score:.4f})")
        else:
            print(f"No similar terms found for '{query}'.")

if __name__ == "__main__":
    main()
