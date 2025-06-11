# EVE-RAG
This repository contains code and experiments for a Retrieval-Augmented Generation (RAG) pipeline. The goal is to explore techniques to improve question answering and reasoning by augmenting LLMs with external knowledge sources.

## Resources
- Rag Survey 2023: https://arxiv.org/abs/2312.10997 
- Rag Survey 2024: https://arxiv.org/abs/2402.19473 
- Rag survey github: https://github.com/Tongji-KGLLM/RAG-Survey
- Rag model first proposed by Facebook: https://arxiv.org/pdf/2005.11401.pdf
- PaperQA - how to do high-accuracy RAG with scientific papers: https://github.com/Future-House/paper-qa


## Suggested Branching Strategy: GitFlow
Use [GitFlow](https://danielkummer.github.io/git-flow-cheatsheet/) for structured branching and release management. This works especially well for ML projects that evolve through cycles of experiments, validations, and deployments.

GitFlow Branches:
- main: Production-ready state. 
- develop: Active development (new features, experiments).
- feature/*: New features, models, or experiments.
- release/*: Preparation for a stable release (e.g., paper submission, benchmark). 
- hotfix/*: Urgent fixes to production code.