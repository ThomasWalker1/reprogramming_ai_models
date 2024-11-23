from goodfire import Client
import goodfire
import json
from tqdm import tqdm

with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])

with open("few_shot_questions.json","r") as file:
    questions=json.load(file)

variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

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
    with open("few_shot_questions.json","w",encoding="utf-8") as file:
        json.dump(questions,file)