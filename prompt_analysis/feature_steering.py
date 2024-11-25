from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import pandas as pd


def query_model(variant,prompt):
    completion=""
    for token in client.chat.completions.create([{"role": "user", "content": prompt}],model=variant,stream=True,max_completion_tokens=150,top_p=0.05):
        completion+=token.choices[0].delta.content
    return completion

GOODFIRE_API_KEY = ""

client = goodfire.Client(
    GOODFIRE_API_KEY
  )

# Instantiate a model variant
variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")


with open("steering_thresholds.json","r") as file:
    steering_thresholds=json.load(file)
    
    
test_dataset = pd.read_csv("../datasets/math_qa/math_qa_with_answers.csv")
test_dataset_sample = test_dataset.sample(100)


variant.reset()
base_variant = variant
for key in steering_thresholds:
    value = steering_thresholds[key]
    searched_features, relevance = client.features.search(
    key,
    model=variant,
    top_k=1)
    steering_feature = searched_features[0]
    
    variant.reset()
    print(steering_feature,value)
    variant.set(steering_feature, value)
    print(variant)
    
    test_dataset_sample[f'{key}_variant'] = test_dataset_sample['Problem'].apply(lambda x: query_model(variant,x))
    
    
base_variant.reset()
print(base_variant)
test_dataset_sample['base_variant_dataset_1'] = test_dataset_sample['Problem'].apply(lambda x: query_model(base_variant,x))