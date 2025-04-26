def reduce_jsonl_size(input_file, output_file, sample_fraction=0.1):
    import random
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Sample a fraction of lines randomly
    sample_size = int(len(lines) * sample_fraction)
    sampled_lines = random.sample(lines, sample_size)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(sampled_lines)

    print(f"Reduced file saved to {output_file} with {sample_size} lines.")

reduce_jsonl_size('train.jsonl', 'small_train.jsonl', 0.3)  # 30% sample