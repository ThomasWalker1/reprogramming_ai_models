import csv
import json

input_file = 'math_qa_with_answers.csv'
output_file = 'few_shot_mathqa.json'

data = []

with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# Create few-shot examples
few_shot_examples = []
for i in range(0, len(rows), 3):
    example = {"id": i // 3 + 1, "questions": []}
    for j in range(3):
        if i + j < len(rows):
            question = rows[i + j]['Problem']
            answer = rows[i + j]['answer'] if j < 2 else ""
            example["questions"].append({"question": question, "answer": answer})
    few_shot_examples.append(example)

# Convert to JSON format
with open(output_file, mode='w', encoding='utf-8') as outfile:
    json.dump(few_shot_examples, outfile, indent=2)

print(f"Few-shot prompt saved to {output_file}")