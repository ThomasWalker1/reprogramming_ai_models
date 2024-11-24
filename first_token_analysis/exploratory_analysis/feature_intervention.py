from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import os
import pickle

with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant=goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

features,relevance=client.features.search("Introduction of variables or assumptions in mathematical reasoning",model=variant,top_k=5)
print(features[0])
method_num=3

if os.path.exists("methods"):
    with open("methods","rb") as file:
        methods=pickle.load(file)
else:
    methods={}

methods[method_num]={"feature":features[0],
                     "mode":"nudge",
                     "value":0.6}

with open("methods","wb") as file:
    pickle.dump(methods,file)

with open("methods.txt","w") as file:
    for key in sorted(methods.keys()):
        file.write(str(key)+"\n")
        file.write(f"    {methods[key]["feature"]}\n")
        file.write(f"    {methods[key]["mode"]}\n")
        file.write(f"    {methods[key]["value"]}\n")

variant.reset()
variant.set(**methods[method_num])

with open(f"few_shot_questions_method{method_num}.json","r",encoding="utf-8") as file:
    questions=json.load(file)

def query_model(prompt):
    completion=""
    for token in client.chat.completions.create([{"role": "user", "content": prompt}],model=variant,stream=True,max_completion_tokens=128,top_p=0.05):
        completion+=token.choices[0].delta.content
    return completion

pbar=tqdm(enumerate(questions),total=len(questions))
for (k,question) in pbar:
    if not("cot_answer" in question.keys()):
        questions[k]["cot_answer"]=query_model(question["cot"]+"\n"+question["question"])
    if not("direct_answer" in question.keys()):
        questions[k]["direct_answer"]=query_model(question["direct"]+"\n"+question["question"])
    with open(f"few_shot_questions_method{method_num}.json","w",encoding="utf-8") as file:
        json.dump(questions,file)