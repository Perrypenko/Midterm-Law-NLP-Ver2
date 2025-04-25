from TF_IDF import retrieve_relevant_legislation

# Test cases
test_queries = [
    "workplace discrimination",
    "tax evasion penalties",
    "housing regulations for landlords"
]
print("Starting test...")
for query in test_queries:
    print(f"\nTesting query: '{query}'")
    results = retrieve_relevant_legislation(query)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['year']} (Similarity: {result['similarity']:.4f})")
        print(f"   ID: {result['id']}")
        print(f"   Preview: {result['text_preview']}")
print("Test completed.")