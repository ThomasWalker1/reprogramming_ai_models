from goodfire import Client
import goodfire
import json
import os
from tqdm import tqdm
import random

with open("key.txt", "r") as file:
    client = Client(api_key=file.readlines()[0])

variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")



def query_model(prompt):
    completion = ""
    for token in client.chat.completions.create(
            [{"role": "user", "content": prompt}],
            model=variant,
            stream=True,
            max_completion_tokens=512,
            top_p=0.05):
        completion += token.choices[0].delta.content
    return completion



def get_samples_from_category(category_path, num_samples=5):
    print(f"Getting samples from category: {category_path}")
    # Get all JSON files and sort them alphabetically so we can all get the same questions every time
    question_files = sorted([f for f in os.listdir(category_path) if f.endswith('.json')])
    # Take the first num_samples files
    print(f"Total files: {len(question_files)}")
    selected_files = question_files[:num_samples]
    print(f"Selected files: {selected_files}")
    
    samples = []
    for file_name in selected_files:
        print(f"Processing file: {file_name}")
        with open(os.path.join(category_path, file_name), 'r', encoding='utf-8') as f:
            print(f"Opened file: {file_name}")
            data = json.load(f)
            sample = {
                "problem_text": data["problem"],
                "reference_answer": data["solution"],
                "model_answer": query_model(f"Solve this math problem: {data['problem']}")
            }
            samples.append(sample)
            print(f"Generated sample: {sample}")
    return samples


categories = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]

all_samples = []
#pbar = tqdm(categories, desc="Processing categories")
for category in categories:
    category_path = os.path.join('MATH', 'train', category)
    print(f"Processing category: {category}")
    category_samples = get_samples_from_category(category_path)
    print(f"Generated {len(category_samples)} samples for category: {category}")
    
    print(f"Samples: {category_samples}")
    for sample in category_samples:
        sample["category"] = category
    all_samples.extend(category_samples)
    
    # Save progress after each category
    with open("math_samples_with_answers.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

print(f"Completed! Generated {len(all_samples)} samples across {len(categories)} categories.")


