from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import pickle
import os
from copy import copy
from scipy.stats import skew

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

def get_data():
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

answer_validity={
    1:{"cot": True,
       "direct": True},
    2:{"cot": True,
       "direct": False},
    3:{"cot": True,
       "direct": False},
    4:{"cot": True,
       "direct": False},
    5:{"cot": True,
       "direct": True},
    6:{"cot": True,
       "direct": True},
    7:{"cot": True,
       "direct": True},
    8:{"cot": True,
       "direct": True},
    9:{"cot": True,
       "direct": False},
    10:{"cot": True,
       "direct": True},
    11:{"cot": True,
       "direct": False},
    12:{"cot": True,
       "direct": False},
    13:{"cot": True,
       "direct": False},
    14:{"cot": True,
       "direct": True},
    15:{"cot": True,
       "direct": True},
    16:{"cot": True,
       "direct": False},
    17:{"cot": True,
       "direct": True},
    18:{"cot": False,
       "direct": False},
    19:{"cot": True,
       "direct": True},
    20:{"cot": True,
       "direct": True}
}

def summarise_data():
    with open("features_at_first_token","rb") as file:
        features_at_first_token=pickle.load(file)
    
    cot_features={}
    direct_features={}

    for answer_id in features_at_first_token.keys():
        if answer_validity[answer_id]["cot"]:
            for (act,feat) in features_at_first_token[answer_id]["cot_features"].items():
                if feat[0] in cot_features.keys():
                    cot_features[feat[0]]["activations"].append(act)
                else:
                    cot_features[feat[0]]={"activations":[act],
                                           "description":feat[1]}
        if answer_validity[answer_id]["direct"]:
            for (act,feat) in features_at_first_token[answer_id]["direct_features"].items():
                if feat[0] in direct_features.keys():
                    direct_features[feat[0]]["activations"].append(act)
                else:
                    direct_features[feat[0]]={"activations":[act],
                                           "description":feat[1]}
    cot_feature_counts={feat_id:len(vals["activations"]) for (feat_id,vals) in cot_features.items()}
    direct_feature_counts={feat_id:len(vals["activations"]) for (feat_id,vals) in direct_features.items()}
    cot_feature_cum={feat_id:sum(vals["activations"]) for (feat_id,vals) in cot_features.items()}
    direct_feature_cum={feat_id:sum(vals["activations"]) for (feat_id,vals) in direct_features.items()}

    cot_activation_skewness=[]
    direct_activation_skewness=[]

    for answer_id in features_at_first_token.keys():
        if answer_validity[answer_id]["cot"]:
            cot_activation_skewness.append(skew(list(features_at_first_token[answer_id]["cot_features"].keys())))
        if answer_validity[answer_id]["direct"]:
            direct_activation_skewness.append(skew(list(features_at_first_token[answer_id]["direct_features"].keys())))


    with open("features_at_first_token_summary.txt","w") as file:
        file.write(f"CoT Activation Skewness - {sum(cot_activation_skewness)/len(cot_activation_skewness)}\n")
        file.write(f"Direct Activation Skewness - {sum(direct_activation_skewness)/len(direct_activation_skewness)}\n")
        file.write("\nCoT Features by Count\n")
        for (feat_id,count) in dict(sorted(cot_feature_counts.items(), key=lambda item: item[1],reverse=True)).items():
            file.write(f"    {count} - {feat_id} - {cot_features[feat_id]["description"]}\n")
        file.write("Direct Features by Count\n")
        for (feat_id,count) in dict(sorted(direct_feature_counts.items(), key=lambda item: item[1],reverse=True)).items():
            file.write(f"    {count} - {feat_id} - {direct_features[feat_id]["description"]}\n")

        file.write("\nCoT Features by Cumulative Activation\n")
        for (feat_id,count) in dict(sorted(cot_feature_cum.items(), key=lambda item: item[1],reverse=True)).items():
            file.write(f"    {count} - {feat_id} - {cot_features[feat_id]["description"]}\n")
        file.write("Direct Features by Cumulative Activation\n")
        for (feat_id,count) in dict(sorted(direct_feature_cum.items(), key=lambda item: item[1],reverse=True)).items():
            file.write(f"    {count} - {feat_id} - {direct_features[feat_id]["description"]}\n")

    


summarise_data()