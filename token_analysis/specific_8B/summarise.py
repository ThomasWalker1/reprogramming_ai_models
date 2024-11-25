import json

methods=["baseline","enthusiasm_steering","mathematical_reasoning_steering_nudge","mathematical_reasoning_steering","step_by_step_reasoning"]
for method in methods:
    with open(f"first_token_analysis/specific_analysis/{method}.json","r",encoding="utf-8") as file:
        responses=json.load(file)

    with open(f"first_token_analysis/specific_analysis/{method}.txt","w",encoding="utf-8") as file:
        for response in responses:
            file.write(f"response {response["id"]}\n")
            file.write(f"    {response["response"].replace("\n","<nl>")}\n")
