import json
methods=["baseline","e_s_steering","m_e_steering","m_s_steering","m_s_e_steering"]
lengths={}
for method in methods:
    with open(f"first_token_analysis/final_analysis/{method}.json","r",encoding="utf-8") as file:
        responses=json.load(file)
    lengths[method]=[len(response["response"]) for response in responses]

with open(f"first_token_analysis/final_analysis/response_lengths.txt","w",encoding="utf-8") as file:
    for (method,lengths) in lengths.items():
        file.write(f"Method {method}\n")
        file.write(f"    mean - {sum(lengths)/len(lengths)}\n")
        file.write(f"    raw - {lengths}\n")
