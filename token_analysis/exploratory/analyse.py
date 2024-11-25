from goodfire import Client
import goodfire
import json
from tqdm import tqdm
import pickle
import os
from scipy.stats import skew

# Set up and load in the model variant from Goodfire
with open("key.txt","r") as file:
    client=Client(api_key=file.readlines()[0])
variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

# wrapper function to extract the top eight features at the first token given by the assistant
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

# wrapper function to collect the features of the baseline responses
def get_data():

    # dictionary to store the extracted features
    if os.path.exists("features"):
        with open("features","rb") as file:
            features=pickle.load(file)
    else:
        features={}

    # load in the baseline responses
    with open("token_analysis/exploratory/responses_baseline.json","r",encoding="utf-8") as file:
        responses=json.load(file)

    pbar=tqdm(responses,total=len(responses))
    for response in pbar:
        q_id=response["id"]
        if q_id in features.keys():
            continue
        
        # extract the features at the CoT and direct answers
        features[q_id]={}
        if "cot_answer" in response.keys():
            cot_features,tok=top_features(response["cot"]+"\n"+response["question"],response["cot_answer"][:32])
            features[q_id]["cot_features"]=cot_features
            features[q_id]["cot_token"]=tok
        if "direct_answer" in response.keys():
            direct_features,tok=top_features(response["direct"]+"\n"+response["question"],response["direct_answer"][:32])
            features[q_id]["direct_features"]=direct_features
            features[q_id]["direct_token"]=tok

        # display the features for easy analysis
        with open("token_analysis/exploratory/features.txt","w") as file:
            for id in sorted(features.keys()):
                file.write(str(id)+"\n")
                if "cot_features" in features[id]:
                    file.write(f"  chain-of-thought features at {features[id]["cot_token"]}\n")
                    for activation in sorted(features[id]["cot_features"].keys(),reverse=True):
                        file.write(f"    {features[id]["cot_features"][activation]} {activation}\n")
                if "direct_features" in features[id]:
                    file.write(f"  direct features at {features[id]["direct_token"]}\n")
                    for activation in sorted(features[id]["direct_features"].keys(),reverse=True):
                        file.write(f"    {features[id]["direct_features"][activation]} {activation}\n")

    # save the collection of features
    with open("token_analysis/exploratory/features","wb") as file:
        pickle.dump(features,file)

# here we check through the given responses and determine which ones
# exhibit the desired behavior (i.e. give CoT reason or answer directly
# when prompted to do so).
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

# wrapper function to analyse the features
def summarise_data():
    with open("token_analysis/exploratory/features","rb") as file:
        features=pickle.load(file)
    
    cot_features={}
    direct_features={}

    for answer_id in features.keys():
        if answer_validity[answer_id]["cot"]:
            for (act,feat) in features[answer_id]["cot_features"].items():
                if feat[0] in cot_features.keys():
                    cot_features[feat[0]]["activations"].append(act)
                else:
                    cot_features[feat[0]]={"activations":[act],
                                           "description":feat[1]}
        if answer_validity[answer_id]["direct"]:
            for (act,feat) in features[answer_id]["direct_features"].items():
                if feat[0] in direct_features.keys():
                    direct_features[feat[0]]["activations"].append(act)
                else:
                    direct_features[feat[0]]={"activations":[act],
                                           "description":feat[1]}
    # counting the occurrence of features
    cot_feature_counts={feat_id:len(vals["activations"]) for (feat_id,vals) in cot_features.items()}
    direct_feature_counts={feat_id:len(vals["activations"]) for (feat_id,vals) in direct_features.items()}
    # counting the cumulative activation of features
    cot_feature_cum={feat_id:sum(vals["activations"]) for (feat_id,vals) in cot_features.items()}
    direct_feature_cum={feat_id:sum(vals["activations"]) for (feat_id,vals) in direct_features.items()}

    # calculating the skewness of the activation distribution of the features
    cot_activation_skewness=[]
    direct_activation_skewness=[]

    for answer_id in features.keys():
        if answer_validity[answer_id]["cot"]:
            cot_activation_skewness.append(skew(list(features[answer_id]["cot_features"].keys())))
        if answer_validity[answer_id]["direct"]:
            direct_activation_skewness.append(skew(list(features[answer_id]["direct_features"].keys())))


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