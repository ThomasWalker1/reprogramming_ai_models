# Reprogramming AI Models Apart Research Hackathon

This is the repository for our project "Encouraging Chain-of-Thought Reasoning in Language Models Through Feature Steering". The project was conducted by Soumyadeep Bose, Kutay Buyruk, Shreyan Jain and I (Thomas Walker), as part of the Apart Research spring "Reprogramming AI Models" sponsored by Goodfire AI.

The report for our project can be found [here.](report.pdf)

## Motivation

We identify active features during chain-of-thought reasoning with the intention to then use them to steer the model to conduct chain-of-thought reasoning. Although most large language models answer questions using chain-of-thought reasoning by default - since they are fine-tuned on vast amounts of this data - there are a few motivations of why identify these features is desirable.
- Chain-of-thought reasoning is not really understood from a interpretability perspective. To our knowledge there only exists works (1) and (2) applying methods of interpretability of chain-of-thought reasoning. Since chain-of-thought reasoning can improve model performance, it seems natural to want to understand how it works.
- Typically, chain-of-thought reasoning in models is elicited with heavy prompt engineering. Consequently, its responses are heavily influenced by the user, and thus one cannot get a true sense of what the model is capable of. Therefore, through steering techniques we can better understand the reasoning capabilities of the model. Similarly, feature steering provides a principled and systematic method for inducing chain-of-thought reasoning, which heavy prompt engineering lacks.
- Our intention is that by steering the model toward reasoning with chain-of-thought, get more effective chain-of-thought reasoning. That is, the reasoning is more coherent, succinct and accurate; allowing us to better interpret what the model is thinking.
- Moreover, since chain-of-thought reasoning can significantly influence model capabilities, having the ability to directly control this behavior through feature steering means we can have greater control over the capabilities of these large language models. This is important for a model deployer, who cannot always guarantee their models will be deployed in low-risk environments.


**References**

(1) Wei, J. et al. (2023) ‘Chain-of-Thought Prompting Elicits Reasoning in Large Language Models’. arXiv. Available at: https://doi.org/10.48550/arXiv.2201.11903.

(2) Wu, S. et al. (2023) ‘Analyzing Chain-of-Thought Prompting in Large Language Models via Gradient-based Feature Attributions’. arXiv. Available at: https://doi.org/10.48550/arXiv.2307.13339.