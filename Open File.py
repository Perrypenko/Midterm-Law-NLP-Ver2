import json

# Define the path to your JSONL file
jsonl_path = 'uk_legislation/validation.jsonl'

# Function to load a few lines from the JSONL file to inspect structure
def load_sample_jsonl(file_path, num_lines=5):
    sample_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            sample_data.append(json.loads(line))
    return sample_data

# Load sample data
try:
    sample_legislation = load_sample_jsonl(jsonl_path)
    
    # Print keys of the first few records to understand the structure
    for i, record in enumerate(sample_legislation):
        print(f"Record {i+1} keys: {list(record.keys())}")
        print(f"Sample data: {record}\n")
except FileNotFoundError:
    print(f"File not found: {jsonl_path}")
    print("Make sure the file exists in the specified location.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("Check if the file is in valid JSONL format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Please check the file and try again.")