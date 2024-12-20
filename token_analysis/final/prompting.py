from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import os
import pickle

# Set up and load in the model variant from Goodfire
with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant=goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")

# identify the method of intervention
method="m_s_e_steering"

if "baseline" in method:
    variant.reset()
elif "steering" in method:
    # extract the relevant features
    e_features,relevance=client.features.search("The model's turn to speak, especially with enthusiasm",model=variant,top_k=5)
    s_features,relevance=client.features.search("The model is beginning a step-by-step explanation",model=variant,top_k=5)
    m_features,relevance=client.features.search("Introduction of variables or assumptions in mathematical reasoning",model=variant,top_k=5)

    # open file to record methods
    if os.path.exists("token_analysis/final/methods"):
        with open("token_analysis/final/methods","rb") as file:
            methods=pickle.load(file)
    else:
        methods={}

    # set the intervention strategy
    methods[method]={"feature":[m_features[0],s_features[0],e_features[0]],
                     "mode":"pin",
                     "value":0.6}
    variant.set(**methods[method])

    # record the method
    with open("token_analysis/final/methods","wb") as file:
        pickle.dump(methods,file)

    with open("token_analysis/final/methods.txt","w") as file:
        for key in sorted(methods.keys()):
            file.write(key+"\n")
            file.write(f"    {methods[key]["feature"]}\n")
            file.write(f"    {methods[key]["mode"]}\n")
            file.write(f"    {methods[key]["value"]}\n")

# load in the sample of questions
with open(f"../datasets/mathqa_24sample.json","r",encoding="utf-8") as file:
    questions=json.load(file)

# wrapper function to query the model and return the contents of its response
def query_model(prompt):
    completion=""
    for token in client.chat.completions.create([{"role": "user", "content": prompt}],model=variant,stream=True,max_completion_tokens=1024,top_p=0,temperature=0):
        completion+=token.choices[0].delta.content
    return completion

pbar=tqdm(enumerate(questions),total=len(questions))
for (k,data) in pbar:
    if not("response" in data.keys()):

        prompt=""
        for q_data in data["questions"]:
            prompt+="Answer the following question.\n"
            prompt+=f"Q: {q_data["question"]}\nA: "

        questions[k]["prompt"]=prompt
        questions[k]["response"]=query_model(prompt)

# record the model responses
with open(f"token_analysis/final/{method}.json","w",encoding="utf-8") as file:
    json.dump(questions,file)