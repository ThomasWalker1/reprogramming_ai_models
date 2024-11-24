from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import os
import pickle

with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant=goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")

method="step_by_step_high_pin"

if "baseline" in method:
    variant.reset()
elif "steering" in method:
    e_features,relevance=client.features.search("The model's turn to speak, especially with enthusiasm",model=variant,top_k=5)
    s_features,relevance=client.features.search("The model is beginning a step-by-step explanation",model=variant,top_k=5)
    m_features,relevance=client.features.search("Introduction of variables or assumptions in mathematical reasoning",model=variant,top_k=5)

    if os.path.exists("first_token_analysis/specific_analysis_70B/methods"):
        with open("first_token_analysis/specific_analysis_70B/methods","rb") as file:
            methods=pickle.load(file)
    else:
        methods={}

    methods[method]={"feature":s_features[0],
                     "mode":"pin",
                     "value":0.8}

    variant.set(**methods[method])


    with open("first_token_analysis/specific_analysis_70B/methods","wb") as file:
        pickle.dump(methods,file)

    with open("first_token_analysis/specific_analysis_70B/methods.txt","w") as file:
        for key in sorted(methods.keys()):
            file.write(key+"\n")
            file.write(f"    {methods[key]["feature"]}\n")
            file.write(f"    {methods[key]["mode"]}\n")
            file.write(f"    {methods[key]["value"]}\n")


with open(f"first_token_analysis/specific_analysis_70B/{method}.json","r",encoding="utf-8") as file:
    questions=json.load(file)

def query_model(prompt):
    completion=""
    for token in client.chat.completions.create([{"role": "user", "content": prompt}],model=variant,stream=True,max_completion_tokens=512,top_p=0,temperature=0):
        completion+=token.choices[0].delta.content
    return completion

pbar=tqdm(enumerate(questions),total=len(questions))
for (k,data) in pbar:
    if not("response" in data.keys()):

        #prompt="Here are some examples of questions and answers.\n"
        prompt=""
        for q_data in data["questions"]:
            #if q_data["answer"]!="":
                #prompt+=f"Q: {q_data["question"]}\nA: {q_data["answer"]}\n"
            #else:
            if q_data["answer"]=="":
                #prompt+="Now answer the following question.\n"
                prompt+="Answer the following question.\n"
                prompt+=f"Q: {q_data["question"]}\nA: "

        questions[k]["prompt"]=prompt
        questions[k]["response"]=query_model(prompt)


with open(f"first_token_analysis/specific_analysis_70B/{method}.json","w",encoding="utf-8") as file:
    json.dump(questions,file)