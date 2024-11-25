from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import os
import pickle

# Set up and load in the model variant from Goodfire
with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant=goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

# Record the method of feature intervention we are investigating
method_num=3
if os.path.exists("methods"):
    with open("methods","rb") as file:
        methods=pickle.load(file)
else:
    methods={}

features,relevance=client.features.search("Introduction of variables or assumptions in mathematical reasoning",model=variant,top_k=5)
methods[method_num]={"feature":features[0],"mode":"nudge","value":0.6}

with open("methods","wb") as file:
    pickle.dump(methods,file)

with open("methods.txt","w") as file:
    for key in sorted(methods.keys()):
        file.write(str(key)+"\n")
        file.write(f"    {methods[key]["feature"]}\n")
        file.write(f"    {methods[key]["mode"]}\n")
        file.write(f"    {methods[key]["value"]}\n")

# Set the method of feature intervention onto the variant
variant.reset()
variant.set(**methods[method_num])

# load in the questions
with open(f"responses_{method_num}.json","r",encoding="utf-8") as file:
    questions=json.load(file)

# wrapper function to query the model and return the contents of its response
def query_model(prompt):
    completion=""
    for token in client.chat.completions.create([{"role": "user", "content": prompt}],model=variant,stream=True,max_completion_tokens=128,top_p=0.05):
        completion+=token.choices[0].delta.content
    return completion

# iterate through the questions
pbar=tqdm(enumerate(questions),total=len(questions))
for (k,question) in pbar:
    if not("cot_answer" in question.keys()):
        # getting the model response when few-shotted to give an answers using chain-of-thought reasoning
        questions[k]["cot_answer"]=query_model(question["cot"]+"\n"+question["question"])
    if not("direct_answer" in question.keys()):
        # getting the model response when few-shotted to give an using directly
        questions[k]["direct_answer"]=query_model(question["direct"]+"\n"+question["question"])
    with open(f"few_shot_questions_method{method_num}.json","w",encoding="utf-8") as file:
        json.dump(questions,file)