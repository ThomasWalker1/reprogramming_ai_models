from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import pickle
import os
from copy import copy

with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

with open("few_shot_questions.json","r",encoding="utf-8") as file:
    questions=json.load(file)

if os.path.exists("features_at_first_token"):
    with open("features_at_first_token","rb") as file:
        features_at_first_token=pickle.load(file)
else:
    features_at_first_token={}

def top_features(user,assistant):
    variant.reset()

    context=client.features.inspect([
        {"role": "user","content": user},
        {"role": "assistant","content": assistant},],
        model=variant,)
    
    toks=context.tokens
    for n in range(len(toks)-1):
        if str(toks[n])=='Token("assistant")' and str(toks[n+1])=='Token("<|end_header_id|>")':
            start_tok_idx=n+3
    feats=context.tokens[start_tok_idx].inspect(k=8)
    s_vector,feat_lookup=feats.vector()
    feat_data={}
    for feat_id in feat_lookup:
        feat_data[s_vector[feat_id]]=(feat_id,feat_lookup[feat_id])
    return feat_data,toks[start_tok_idx]

pbar=tqdm(questions,total=len(questions))
for question in pbar:
    q_id=question["id"]
    if q_id in features_at_first_token.keys():
        continue
    
    features_at_first_token[q_id]={}
    if "cot_answer" in question.keys():
        cot_features,tok=top_features(question["cot"]+"\n"+question["question"],question["cot_answer"][:32])
        features_at_first_token[q_id]["cot_features"]=cot_features
        features_at_first_token[q_id]["cot_token"]=tok
    if "direct_answer" in question.keys():
        direct_features,tok=top_features(question["direct"]+"\n"+question["question"],question["direct_answer"][:32])
        features_at_first_token[q_id]["direct_features"]=direct_features
        features_at_first_token[q_id]["direct_token"]=tok

    with open("features_at_first_token.txt","w") as file:
        for id in sorted(features_at_first_token.keys()):
            file.write(str(id)+"\n")
            if "cot_features" in features_at_first_token[id]:
                file.write(f"  chain-of-thought features at {features_at_first_token[id]["cot_token"]}\n")
                for activation in sorted(features_at_first_token[id]["cot_features"].keys(),reverse=True):
                    file.write(f"    {features_at_first_token[id]["cot_features"][activation]} {activation}\n")
            if "direct_features" in features_at_first_token[id]:
                file.write(f"  direct features at {features_at_first_token[id]["direct_token"]}\n")
                for activation in sorted(features_at_first_token[id]["direct_features"].keys(),reverse=True):
                    file.write(f"    {features_at_first_token[id]["direct_features"][activation]} {activation}\n")

with open("features_at_first_token","wb") as file:
    pickle.dump(features_at_first_token,file)