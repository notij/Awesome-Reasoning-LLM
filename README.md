# Awesome-Reasoning-Search-Augmented-LLMs

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A curated list of **Reasoning-Search-Augmented Large Language Models**, focusing on methods that synergize **complex reasoning capabilities** (e.g., Chain-of-Thought, Tree-of-Thoughts, Self-Reflection) with **external information retrieval** (e.g., Web Search, Tool Use, Knowledge Graphs).

This list covers the evolution from standard Retrieval-Augmented Generation (RAG) to autonomous **Search Agents** that interleave reasoning steps with dynamic information seeking to solve multi-hop and open-ended problems.

The repository is part of our survey paper **[Reasoning-Search-Augmented LLMs: A Comprehensive Survey](Paper)** and will be continuously updated.

> **NOTE 1:** To avoid ambiguity, we distinguish between **"Standard RAG"** (Retrieve-then-Generate) and **"Reasoning-Search"** (Interleaved/Agentic). This list focuses on models where the retrieval process is actively controlled by the model's reasoning process (e.g., deciding *when* to search and *what* to search for).

> **NOTE 2:** In each subsection, papers are sorted chronologically. If a paper has a preprint (e.g., arXiv) version, its publication date is according to the preprint service. Otherwise, it follows the conference proceeding or journal date.

> **NOTE 3:** We appreciate contributions! If you have suggested papers, feel free to reach out to [bpoudel@tamu.edu](mailto:your_email@example.com) or submit a [pull request](https://github.com/notij/Awesome-Reasoning-Search-Augmented-LLMs/pulls). For format consistency, please include: (1) the paper title with author names, (2) the publication venue/date, and (3) links to the PDF/Code.

---

## 🧩 Taxonomy

![Taxonomy Diagram](taxonomy_diagram.png)

We categorize the field into the following key dimensions:
- **Architecture:** Modular (Agent-based) vs. End-to-End Trained
- **Reasoning Flow:** Sequential (Chain) vs. Hierarchical (Tree/Graph)
- **Search Interaction:** Single-step vs. Iterative/Interleaved
- **Feedback Mechanism:** Open-loop vs. Self-Correcting (Closed-loop)

## Contents
- [Reinforcement-Learning-Based](#reinforcement-learning-based)
  - [Single‑Controller RL Agents](#Single-Agents)
  - [Multi‑Agent / Modular RL Frameworks](#Multi‑Agent)
  - [Efficiency and Long‑Horizon RL](#Long-Horizon)
- [Prompt-Based and Supervised Search Agents](#prompt-based-and-supervised-search-agents)
- [Tree‑Search and Hierarchical Planning Approaches](#tree-search-and-hierarchical-planning-approaches)  
- [Retrieval-Augmented Generation Variants](#retrieval-augmented-generation-variants)
- [Knowledge Graph & Structured Retrieval](#knowledge-graph--structured-retrieval)

## 📚 Paper List

### Reinforcement-Learning-Based
<h3 id="Single-Agents">Single‑Controller RL Agents</h3>

- **(Search-R1)** Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning ```COLM 2025```     
[[Paper](https://arxiv.org/abs/2503.09516)] [[GitHub](https://github.com/PeterGriffinJin/Search-R1)] [[Models](https://huggingface.co/collections/PeterJinGo/search-r1)]

- **(ZeroSearch)** ZeroSearch: Incentivize the Search Capability of LLMs without Searching ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2505.04588)] [[GitHub](https://github.com/Alibaba-NLP/ZeroSearch)] [[Models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-google-v1)]

- **(ReSearch)** ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2503.19470)] [[GitHub](https://github.com/Agent-RL/ReCall)] [[Models](https://huggingface.co/collections/agentrl/research)]

- **(StepSearch)** StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization ```EMNLP 2025```     
[[Paper](https://arxiv.org/abs/2505.15107)] [[GitHub](https://github.com/Zillwang/StepSearch)] [[Model (3B Base)](https://huggingface.co/Zill1/StepSearch-3B-Base/tree/main)] [[Model (7B Base)](https://huggingface.co/Zill1/StepSearch-7B-Base/tree/main)] [[Model (3B Instruct)](https://huggingface.co/Zill1/StepSearch-3B-Instruct/tree/main)] [[Model (7B Instruct)](https://huggingface.co/Zill1/StepSearch-7B-Instruct/tree/main)] [[Dataset](https://huggingface.co/datasets/Zill1/StepSearch-musi-dataset)]

- **(Stratified GRPO)** Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2510.06214)] 

- **(AI-SearchPlanner)** AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning ```ICLR 2026 UnderReview```     
[[Paper](https://arxiv.org/abs/2508.20368)] 

- **(O2-Searcher)** O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2505.16582)] [[GitHub](https://github.com/KnowledgeXLab/O2-Searcher)] [[Model (3B)](https://huggingface.co/Jianbiao/O2-Searcher-Qwen2.5-3B-GRPO)] [[Dataset](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/tree/main)]

- **(R-Search)**  R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2506.04185)] [[GitHub](https://github.com/QingFei1/R-Search)] [[Models](https://huggingface.co/collections/qingfei1/r-search)] [[Dataset](https://huggingface.co/datasets/qingfei1/R-Search_datasets)]

- **(R3-RAG)** R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2505.23794)] [[GitHub](https://github.com/Yuan-Li-FNLP/R3-RAG)] [[Model(Llama)](https://huggingface.co/Yuan-Li-FNLP/R3-RAG-Llama)] [[Model(Qwen)](https://huggingface.co/Yuan-Li-FNLP/R3-RAG-Qwen)] [[Dataset](https://huggingface.co/datasets/Yuan-Li-FNLP/R3-RAG-RLTrainingData)]

- **(Thinker)** Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction
 ```AAAI 2026```     
[[Paper](https://arxiv.org/abs/2511.07943)] [[GitHub](https://github.com/OpenSPG/KAG-Thinker)] [[Model (7B)](https://huggingface.co/OpenSPG/KAG-Thinker-en-7b-instruct)] [[Dataset](https://huggingface.co/datasets/OpenSPG/KAG-Thinker-training-dataset)]

- **(MemSearcher)** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2511.02805)] [[GitHub](https://github.com/icip-cas/MemSearcher)] 

- **(R1-Searcher)** R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2503.05592)] [[GitHub](https://github.com/RUCAIBox/R1-Searcher)] [[Model (Qwen)]( https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)] [[Model (Llama)](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL)] [[Dataset](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)]

- **(KunLunBaizeRAG)** KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2506.19466)] 

- **(RAG-R1)** RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2507.02962)] [[GitHub](https://github.com/inclusionAI/AWorld-RL/tree/main/RAG-R1)] [[Models](https://huggingface.co/collections/endertzw/rag-r1)]

- **(ReSum)** ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2509.13313)] [[GitHub](https://github.com/Alibaba-NLP/DeepResearch)] [[Model (30B)](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)] [[Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)]

- **(SEM)** SEM: Reinforcement Learning for Search-Efficient Large Language Models ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2505.07903)] 

- **(s3)** s3: You Don't Need That Much Data to Train a Search Agent via RL ```EMNLP 2025```     
[[Paper](https://arxiv.org/abs/2505.14146)] [[GitHub](https://github.com/pat-jj/s3)] [[Dataset](https://huggingface.co/datasets/pat-jj/s3_processed_data)] 

- **(ToolRL)** ToolRL: Reward is All Tool Learning Needs ```NeurIPS 2025 poster```     
[[Paper](https://arxiv.org/abs/2504.13958)] [[GitHub](https://github.com/qiancheng0/ToolRL)] [[Models](https://huggingface.co/collections/emrecanacikgoz/toolrl)] [[Dataset](https://github.com/qiancheng0/ToolRL/tree/main/dataset)]

- **(InForage)** Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging ```NeurIPS 2025```     
[[Paper](https://arxiv.org/abs/2505.09316)] [[GitHub]()] [[Models]()]

- **(HRPO)** Hybrid Latent Reasoning via Reinforcement Learning ```NeurIPS 2025```     
[[Paper](https://arxiv.org/abs/2505.18454)] [[GitHub](https://github.com/Yueeeeeeee/HRPO)]



<h3 id="Multi‑Agent">Multi‑Agent / Modular RL Frameworks</h3>

- **(VerlTool)** VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2509.01055)] [[GitHub](https://github.com/TIGER-AI-Lab/verl-tool)] 

- **(MemSearcher)** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2511.02805)] [[GitHub](https://github.com/icip-cas/MemSearcher)] 

- **(R1-Searcher)** R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2503.05592)] [[GitHub](https://github.com/RUCAIBox/R1-Searcher)] 

- **(CoRAG)** Chain-of-Retrieval Augmented Generation ```NeurIPS 2025```     
[[Paper](https://arxiv.org/abs/2501.14342)] [[GitHub](https://github.com/microsoft/LMOps/tree/main/corag)] [[Model (8B)](https://huggingface.co/corag/CoRAG-Llama3.1-8B-MultihopQA)]

- **(DeepResearcher)** DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments ```EMNLP 2025```     
[[Paper](https://arxiv.org/abs/2504.03160)] [[GitHub](https://github.com/GAIR-NLP/DeepResearcher)] [[Model (7B)](https://huggingface.co/GAIR/DeepResearcher-7b)]

- **(SmartRAG)** SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback ```ICLR 2025```     
[[Paper](https://arxiv.org/abs/2410.18141)] [[GitHub](https://github.com/gaojingsheng/SmartRAG)] [[Model (7B)](https://github.com/AkariAsai/self-rag)]



<h3 id="Long-Horizon">Efficiency and Long‑Horizon RL</h3>

- **(ASearcher)** Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2508.07976)] [[GitHub](https://github.com/inclusionAI/ASearcher)] [[Models](https://huggingface.co/collections/inclusionAI/asearcher)] [[Dataset](https://huggingface.co/datasets/inclusionAI/ASearcher-train-data)]

- **(ReSum)** ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2509.13313)] [[GitHub](https://github.com/Alibaba-NLP/DeepResearch)] [[Model (30B)](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)] [[Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)]

- **(SEM)** SEM: Reinforcement Learning for Search-Efficient Large Language Models ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2509.08827)] 

- **(s3)** s3: You Don't Need That Much Data to Train a Search Agent via RL ```EMNLP 2025```     
[[Paper](https://arxiv.org/abs/2505.14146)] [[GitHub](https://github.com/pat-jj/s3)] [[Dataset](https://huggingface.co/datasets/pat-jj/s3_processed_data)]

- **(MemSearcher)** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2511.02805)] [[GitHub](https://github.com/icip-cas/MemSearcher)] 

### Prompt-Based and Supervised Search Agents
- **(Search-o1)** Search-o1: Agentic Search-Enhanced Large Reasoning Models ```EMNLP 2025 (Oral)```     
[[Paper](https://arxiv.org/abs/2501.05366)] [[GitHub](https://github.com/RUC-NLPIR/Search-o1)] [[Project](https://search-o1.github.io/)] 

- **(AutoRefine)** Search and Refine During Think: Facilitating Knowledge Refinement for Improved Retrieval-Augmented Reasoning ```NeurIPS 2025 (Poster)```     
[[Paper](https://arxiv.org/abs/2505.11277)] [[GitHub](https://github.com/syr-cn/AutoRefine)] [[Model (3B)](https://huggingface.co/yrshi/AutoRefine-Qwen2.5-3B-Base)]

- **(DualRAG)** DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering ```ACL 2025```     
[[Paper](https://arxiv.org/abs/2504.18243)] [[GitHub](https://github.com/cbxgss/rag)] 

- **(PAR-RAG)** Credible Plan-Driven RAG Method for Multi-Hop Question Answering ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2504.16787)] 


### Tree-Search and Hierarchical Planning Approaches
- **(THOUGHTSCULPT)** THOUGHTSCULPT: Reasoning with Intermediate Revision and Search ```NAACL 2025```     
[[Paper](https://arxiv.org/abs/2404.05966)] [[GitHub](https://github.com/cyzus/thoughtsculpt)]

- **(RATT)** RATT: A Thought Structure for Coherent and Correct LLM Reasoning ```AAAI 2025```     
[[Paper](https://arxiv.org/abs/2406.02746)] [[GitHub](https://github.com/jinghanzhang1998/RATT)]

- **(ReKG‑MCTS)** ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search ```ACL Findings 2025```     
[[Paper](https://aclanthology.org/2025.findings-acl.484/)] [[GitHub](https://github.com/ShawnKS/rekgmcts)] 

- **(RARE)** RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models ```ACL 2025```     
[[Paper](https://arxiv.org/abs/2412.02830)] [[GitHub](https://github.com/fatebreaker/RARE)]

- **(HTP)** HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking ```ICML 2025```     
[[Paper](https://arxiv.org/abs/2505.02322)] 

- **(LPKG)** Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs ```EMNLP 2024```     
[[Paper](https://arxiv.org/abs/2406.14282)] [[GitHub](https://github.com/zjukg/LPKG)] [[Dataset](https://huggingface.co/datasets/zjukg/LPKG)]

### Retrieval-Augmented Generation Variants
- **(RAG-R1)** RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2507.02962)] [[GitHub](https://github.com/inclusionAI/AWorld-RL/tree/main/RAG-R1)] [[Models](https://huggingface.co/collections/endertzw/rag-r1)]

- **(Open-RAG)** Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models ```EMNLP 2024```     
[[Paper](https://arxiv.org/abs/2410.01782)] [[GitHub](https://github.com/ShayekhBinIslam/openrag)] [[Project](https://openragmoe.github.io/)] [[Model (7B)](https://huggingface.co/shayekh/openrag_llama2_7b_8x135m)] [[Dataset (Train)](https://huggingface.co/datasets/shayekh/openrag_train_data)] [[Dataset (Eval)](https://huggingface.co/datasets/shayekh/openrag_bench)]

- **(KunLunBaizeRAG)** KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2506.19466)]

- **(EviNote-RAG)** EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes ```ICLR 2026 UnderReview```     
[[Paper](https://arxiv.org/abs/2509.00877)] [[GitHub](https://github.com/Da1yuqin/EviNoteRAG)] [[Model (7B)](https://huggingface.co/dayll/EviNoteRAG-7B)] [[Dataset](https://huggingface.co/datasets/dayll/EviNoteRAG_nq_hotpotqa_train_and_test_data)]

- **(SmartRAG)** SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback ```ICLR 2025```     
[[Paper](https://arxiv.org/abs/2410.18141)] [[GitHub](https://github.com/gaojingsheng/SmartRAG)] [[Model (7B)](https://github.com/AkariAsai/self-rag)]

- **(ToolRL)** ToolRL: Reward is All Tool Learning Needs ```NeurIPS 2025 poster```     
[[Paper](https://arxiv.org/abs/2504.13958)] [[GitHub](https://github.com/qiancheng0/ToolRL)] [[Models](https://huggingface.co/collections/emrecanacikgoz/toolrl)] [[Dataset](https://github.com/qiancheng0/ToolRL/tree/main/dataset)]

- **(ZeroSearch)** ZeroSearch: Incentivize the Search Capability of LLMs without Searching ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2505.04588)] [[GitHub](https://github.com/Alibaba-NLP/ZeroSearch)] [[Models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-google-v1)]

- **(R1-Searcher)** R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2503.05592)] [[GitHub](https://github.com/RUCAIBox/R1-Searcher)] [[Model (Qwen)]( https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)] [[Model (Llama)](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL)] [[Dataset](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)]

- **(MemSearcher)** MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2511.02805)] [[GitHub](https://github.com/icip-cas/MemSearcher)] 

### Knowledge Graph & Structured Retrieval
- **(LPKG)** Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs ```EMNLP 2024```     
[[Paper](https://arxiv.org/abs/2406.14282)] [[GitHub](https://github.com/zjukg/LPKG)] [[Dataset](https://huggingface.co/datasets/zjukg/LPKG)]

- **(ReKG‑MCTS)** ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search ```ACL Findings 2025```     
[[Paper](https://aclanthology.org/2025.findings-acl.484/)] [[GitHub](https://github.com/ShawnKS/rekgmcts)] 

- **(PAR-RAG)** Credible Plan-Driven RAG Method for Multi-Hop Question Answering ```arXiv 2025```     
[[Paper](https://arxiv.org/abs/2504.16787)] 

