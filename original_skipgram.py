import json
import numpy as np
from collections import defaultdict, Counter
import random
from sklearn.metrics.pairwise import cosine_similarity

class SkipGramModel:
    def __init__(self, vector_size=100, window_size=2, learning_rate=0.01, epochs=5, min_count=5):
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        self.W = None  # Input -> Hidden weights
        self.W_prime = None  # Hidden -> Output weights
        
    def build_vocab(self, corpus):
        # Count word frequencies
        for sentence in corpus:
            for word in sentence:
                self.word_counts[word] += 1
        
        # Filter words by frequency
        vocab = [word for word, count in self.word_counts.items() if count >= self.min_count]
        
        # Create word mappings
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        # Initialize weights
        self.W = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, 
                                  (self.vocab_size, self.vector_size))
        self.W_prime = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, 
                                        (self.vector_size, self.vocab_size))
    
    def generate_training_data(self, corpus):
        training_data = []
        for sentence in corpus:
            for i, center_word in enumerate(sentence):
                if center_word not in self.word_to_idx:
                    continue
                    
                center_idx = self.word_to_idx[center_word]
                
                # Get context words within window
                context_indices = list(range(max(0, i - self.window_size), i)) + \
                                 list(range(i + 1, min(len(sentence), i + self.window_size + 1)))
                
                for j in context_indices:
                    if j < 0 or j >= len(sentence) or sentence[j] not in self.word_to_idx:
                        continue
                    context_idx = self.word_to_idx[sentence[j]]
                    training_data.append((center_idx, context_idx))
        
        return training_data
    
    def train(self, training_data):
        for epoch in range(self.epochs):
            loss = 0
            random.shuffle(training_data)
            
            for center_idx, context_idx in training_data:
                # Forward pass
                h = self.W[center_idx]  # Hidden layer
                u = np.dot(h, self.W_prime)  # Output layer (unnormalized)
                y_pred = self._softmax(u)  # Predicted probabilities
                
                # Create target vector (one-hot)
                y_true = np.zeros(self.vocab_size)
                y_true[context_idx] = 1
                
                # Compute error
                error = y_pred - y_true
                loss += -np.log(y_pred[context_idx])
                
                # Backpropagation
                # Update output->hidden weights
                self.W_prime -= self.learning_rate * np.outer(h, error)
                
                # Update hidden->input weights
                self.W[center_idx] -= self.learning_rate * np.dot(error, self.W_prime.T)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss/len(training_data)}")
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))  # For numerical stability
        return e_x / e_x.sum()
    
    def get_word_vector(self, word):
        if word in self.word_to_idx:
            return self.W[self.word_to_idx[word]]
        return None
    
    def find_similar_words(self, word, top_n=10):
        if word not in self.word_to_idx:
            print(f"'{word}' not in vocabulary")
            return []
        
        word_vec = self.get_word_vector(word)
        similarities = []
        
        for i in range(self.vocab_size):
            if i != self.word_to_idx[word]:
                other_vec = self.W[i]
                similarity = cosine_similarity([word_vec], [other_vec])[0][0]
                similarities.append((self.idx_to_word[i], similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def load_jsonl(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data:  # Adjust based on your JSONL structure
                # Simple tokenization by splitting on spaces
                tokens = data['text'].lower().split()
                corpus.append(tokens)
    return corpus

def main():
    # File path to your JSONL file
    jsonl_file = "uk_legislation/train.jsonl"
    
    # Load and preprocess the corpus
    print("Loading corpus...")
    corpus = load_jsonl(jsonl_file)
    
    # Initialize and train the Skip-Gram model
    print("Training Skip-Gram model...")
    model = SkipGramModel(vector_size=100, window_size=2, learning_rate=0.01, epochs=5, min_count=5)
    model.build_vocab(corpus)
    training_data = model.generate_training_data(corpus)
    model.train(training_data)
    
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
