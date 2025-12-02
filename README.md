# Awesome-Reasoning-LLM
2025 Fall CSCE-670 Project


## Contents
- [Reinforcement-Learning–Based](#reinforcement-learning-based)
  - [Single‑Controller RL Agents](#Single-Agents)
  - [Multi‑Agent / Modular RL Frameworks](#Multi‑Agent)
  - [Efficiency and Long‑Horizon RL](#Long-Horizon)
- [Prompt-Based and Supervised Search Agents](#promptbased-and-supervised-search-agents)
- [Tree‑Search and Hierarchical Planning Approaches](#Tree-Search) 
- [Retrieval‑Augmented Generation Variants](#Retrival-Augmented)
- [Knowledge Graph & Structured Retrieval](#Knowledge-Graph)
  
## Reinforcement-Learning–Based {#reinforcement-learning-based}
<h3 id="Single-Agents">Single‑Controller RL Agents</h3>

- **(Search-R1)** Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning ```COLM 2025```     
[[Paper](https://arxiv.org/abs/2503.09516)] [[GitHub](https://github.com/PeterGriffinJin/Search-R1)] [[Models](https://huggingface.co/collections/PeterJinGo/search-r1)]

- **(ZeroSearch)** ZeroSearch: Incentivize the Search Capability of LLMs without Searching ```arXiv```     
[[Paper](https://arxiv.org/abs/2505.04588)] [[GitHub](https://github.com/Alibaba-NLP/ZeroSearch)] [[Models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-google-v1)]

- **(ReSearch)** ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning ```arXiv```     
[[Paper](https://arxiv.org/abs/2503.19470)] [[GitHub](https://github.com/Agent-RL/ReCall)] [[Models](https://huggingface.co/collections/agentrl/research)]

- **(StepSearch)** StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization ```EMNLP 2025```     
[[Paper](https://arxiv.org/abs/2505.15107)] [[GitHub](https://github.com/Zillwang/StepSearch)] [[Model (3B Base)](https://huggingface.co/Zill1/StepSearch-3B-Base/tree/main)] [[Model (7B Base)](https://huggingface.co/Zill1/StepSearch-7B-Base/tree/main)] [[Model (3B Instruct)](https://huggingface.co/Zill1/StepSearch-3B-Instruct/tree/main)] [[Model (7B Instruct)](https://huggingface.co/Zill1/StepSearch-7B-Instruct/tree/main)] [[Dataset](https://huggingface.co/datasets/Zill1/StepSearch-musi-dataset)]

- **(Stratified GRPO)** Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents ```arXiv```     
[[Paper](https://arxiv.org/abs/2510.06214)] 

- **(AI-SearchPlanner)** AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning ```ICLR 2026 UnderReview```     
[[Paper](https://arxiv.org/abs/2508.20368)] 

- **(O2-Searcher)** O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering ```arXiv```     
[[Paper](https://arxiv.org/abs/2505.16582)] [[GitHub](https://github.com/KnowledgeXLab/O2-Searcher)] [[Model (3B)](https://huggingface.co/Jianbiao/O2-Searcher-Qwen2.5-3B-GRPO)] [[Dataset](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/tree/main)]

- **(R-Search)**  R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning ```arXiv```     
[[Paper](https://arxiv.org/abs/2506.04185)] [[GitHub](https://github.com/QingFei1/R-Search)] [[Models](https://huggingface.co/collections/qingfei1/r-search)] [[Dataset](https://huggingface.co/datasets/qingfei1/R-Search_datasets)]

- **(R3-RAG)** R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning ```arXiv```     
[[Paper](https://arxiv.org/abs/2505.23794)] [[GitHub](https://github.com/Yuan-Li-FNLP/R3-RAG)] [[Model(Llama)](https://huggingface.co/Yuan-Li-FNLP/R3-RAG-Llama)] [[Model(Qwen)](https://huggingface.co/Yuan-Li-FNLP/R3-RAG-Qwen)] [[Dataset](https://huggingface.co/datasets/Yuan-Li-FNLP/R3-RAG-RLTrainingData)]

- **(Thinker)** Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction
 ```AAAI 2026```     
[[Paper](https://arxiv.org/abs/2511.07943)] [[GitHub](https://github.com/OpenSPG/KAG-Thinker)] [[Model (7B)](https://huggingface.co/OpenSPG/KAG-Thinker-en-7b-instruct)] [[Dataset](https://huggingface.co/datasets/OpenSPG/KAG-Thinker-training-dataset)]

- **(MEMSEARCHER)** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning ```arXiv```     
[[Paper](https://arxiv.org/abs/2511.02805)] [[GitHub](https://github.com/icip-cas/MemSearcher)] 

- **(R1-Searcher)** R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning ```arXiv```     
[[Paper](https://arxiv.org/abs/2503.05592)] [[GitHub](https://github.com/RUCAIBox/R1-Searcher)] [[Model (Qwen)]( https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)] [[Model (Llama)](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL)] [[Dataset](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)]

- **(KunLunBaizeRAG)** KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models ```arXiv```     
[[Paper](https://arxiv.org/abs/2506.19466)] 

- **(RAG-R1)** RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism ```arXiv```     
[[Paper](https://arxiv.org/abs/2507.02962)] [[GitHub](https://github.com/inclusionAI/AWorld-RL/tree/main/RAG-R1)] [[Models](https://huggingface.co/collections/endertzw/rag-r1)]

- **(ReSum)** ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization ```arXiv```     
[[Paper](https://arxiv.org/abs/2509.13313)] [[GitHub](https://github.com/Alibaba-NLP/DeepResearch)] [[Model (30B)](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)] [[Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)]

- **(SEM)** SEM: Reinforcement Learning for Search-Efficient Large Language Models ```arXiv```     
[[Paper](https://arxiv.org/abs/2509.08827)] 

- **(s3)** s3: You Don't Need That Much Data to Train a Search Agent via RL ```EMNLP 2025```     
[[Paper](https://arxiv.org/pdf/2505.14146)] [[GitHub](https://github.com/pat-jj/s3)] [[Dataset](https://huggingface.co/datasets/pat-jj/s3_processed_data)] 

- **(ToolRL)** ToolRL: Reward is All Tool Learning Needs ```NeurIPS 2025 poster```     
[[Paper](https://arxiv.org/abs/2504.13958)] [[GitHub](https://github.com/qiancheng0/ToolRL)] [[Models](https://huggingface.co/collections/emrecanacikgoz/toolrl)] [[Dataset](https://github.com/qiancheng0/ToolRL/tree/main/dataset)]

- **(InForage)** Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging ```NeurIPS 2025```     
[[Paper](https://arxiv.org/abs/2505.09316)] [[GitHub]()] [[Models]()]

- **(HRPO)** Hybrid Latent Reasoning via Reinforcement Learning ```NeurIPS 2025```     
[[Paper](https://arxiv.org/abs/2505.18454)] [[GitHub](https://github.com/Yueeeeeeee/HRPO)]



<h3 id="Multi‑Agent">Multi‑Agent / Modular RL Frameworks</h3>

- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]

<h3 id="Long-Horizon">Efficiency and Long‑Horizon RL</h3>

- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]

## Prompt-Based and Supervised Search Agents {#promptbased-and-supervised-search-agents}
- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]

## Tree‑Search and Hierarchical Planning Approaches {#Tree-Search}
- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]

## Retrieval‑Augmented Generation Variants {#Retrival-Augmented}
- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]

## Knowledge Graph & Structured Retrieval {#Knowledge-Graph}
- **()**  ``` ```     
[[Paper]()] [[GitHub]()] [[Models]()]