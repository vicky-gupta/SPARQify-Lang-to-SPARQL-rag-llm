# Thesis Topic: LLM-Centric Framework for Ontology-Driven SPARQL Query Generation in RAG for DICOM Databases
## Collaborators:
<div align="center">
  <img src="images/faulogo.png" width="250" height="123" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/prlab.png" width="250" height="100" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/sohardlogo.gif" width="250" height="100" />
</div>

> **Transforming natural language questions into precise SPARQL queries using Large Language Models and ontologies.**  
> This repository contains the full codebase, datasets, and experimental resources used in my M.Sc. thesis project at Pattern Recognition Lab in FAU Erlangen-Nürnberg.
> It was the collaboration betwee the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and SOHARD Software GmbH.

---

## HPC Note

> All experiments were originally conducted on an **HPC (High Performance Computing)** system.  
> When reproducing results locally, please **check and update file paths** inside the scripts — some directory structures were altered during data migration.
> **Please feel free to raise the issue or email the author if you are having any problem while reproducing the results.**

---

## Project Overview

This repository contains all assets required to **reproduce and explore** the experiments conducted as part of the thesis:

> *LLM-Centric Framework for Ontology-Driven SPARQL Query Generation in RAG for DICOM Databases.*

The project explores how **Large Language Models (LLMs)** can be fine-tuned and integrated with **RAG (Retrieval-Augmented Generation)** pipelines to generate **ontology-compliant SPARQL queries** from **natural language inputs** — particularly for **DICOM medical imaging databases**.

---
### Table 1 — Summary of Designed Experiments

| **Experiments** | **Details** |
|:----------------|:------------|
| **Experiment 1** | Prompt Engineering + RAG |
| **Experiment 2** | Prompt Engineering + RAG + Fine Tuning |
| **Experiment 3** | Prompt Engineering + RAG + Fine Tuning + Semantic Clues |
| **Experiment 4** | Prompt Engineering + RAG + Fine Tuning + Semantic Clues + OBSR |

---

## Results

**Table 2** presents the performance of four open-source large language models (LLMs): **LLaMA 3.1 8B**, **Olmo 13B**, **Gemma 9B**, and **Mistral 7B**, across four experimental conditions.

We report three metrics:  
- **Correct Queries (Cor)** — proportion of correctly generated queries  
- **F1 Score** — harmonic mean of triple-level precision and recall  
- **BLEU Score** — n-gram overlap with the ground truth  
  
These results quantify how each module — **prompt engineering, retrieval-augmented generation (RAG), fine-tuning, semantic enrichment,** and **OBSR** — impacts query generation quality.

---

### Table 2 - Results of All Experiment Configurations Across LLMs  

| **Model** | **Exp 1 Cor** | **Exp 1 BLEU** | **Exp 1 F1** | **Exp 2 Cor** | **Exp 2 BLEU** | **Exp 2 F1** | **Exp 3 Cor** | **Exp 3 BLEU** | **Exp 3 F1** | **Exp 4 Cor** | **Exp 4 BLEU** | **Exp 4 F1** |
|:-----------|:--------------:|:---------------:|:--------------:|:--------------:|:---------------:|:--------------:|:--------------:|:---------------:|:--------------:|:--------------:|:---------------:|:--------------:|
| **LLaMA 3.1 8B** | 16/50 | 0.080 | 0.240 | 8/50 | 0.014 | 0.105 | 14/50 | 0.030 | 0.166 | 14/50 | 0.030 | 0.166 |
| **Olmo 13B** | 6/50 | 0.177 | 0.159 | 4/50 | 0.149 | 0.112 | 6/50 | 0.154 | 0.126 | 6/50 | 0.154 | 0.126 |
| **Gemma 9B** | **2/50** | 0.216 | 0.222 | 8/50 | 0.187 | 0.173 | 5/50 | 0.199 | 0.184 | 5/50 | 0.199 | 0.184 |
| **Mistral 7B** | 14/50 | 0.254 | 0.271 | 15/50 | 0.254 | 0.280 | **18/50**💕 | **0.289** | **0.311** | 18/50 | 0.289 | 0.311 |

---

### Figure 1 - F1 Score Progression Across Experiments

The following figure illustrates the **F1 Score** progression across all four experiments for each open-source large language model (LLM): **LLaMA 8B**, **Olmo 13B**, **Gemma 9B**, and **Mistral 7B**.

| **Model** | **Exp 1** | **Exp 2** | **Exp 3** | **Exp 4** |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|
| **LLaMA 8B** | 0.2404 | 0.1052 | 0.1666 | 0.1666 |
| **Olmo 13B** | 0.1597 | 0.1127 | 0.1267 | 0.1267 |
| **Gemma 9B** | 0.2222 | 0.1732 | 0.1843 | 0.1843 |
| **Mistral 7B** | 0.2711 | 0.2802 | **0.3117**💕 | **0.3117** |

---

#### Highest Performing Model and Setup

Overall, **Experiment 3**, which combines *prompt engineering*, *RAG retrieval*, *fine-tuning*, and *semantic enrichment*, emerges as the most effective configuration.  

It achieves:
- The **highest accuracy** (36% with Mistral 7B)  
- The **best F1 score**  
- The **highest BLEU score** across multiple models  

The F1 score trends for all experiments and models are illustrated above (see *Figure 1*).

---

## Repository Structure

| Folder/File | Description |
|--------------|-------------|
| **`requirements.txt`** | List of Python dependencies required to reproduce the experiments. |
| **`Experiments_Scripts/`** | Contains all four experiment scripts, including multi-GPU fine-tuning code. |
| **`Datasets/`** | Includes ontology data and fine-tuning datasets used throughout the experiments. |
| **`HPC_Scripts/`** | Contains all SLURM job submission scripts used for HPC-based training and inference. |
| **`Fine_Tuning_Weights/`** | Saved model weights and helper scripts (e.g. `remove_bad_keys.py` for key correction). |
| **`Sparql_Results/`** | Generated SPARQL results and experiment outputs in `.txt` format. |

---

## Datasets and Ontologies

- **`onto_ver_7.owl`** — Base ontology (used as RAG document in Experiment 1).  
- **`onto_ver_7_semantics.owl`** — Enriched ontology version with `rdfs:comment` annotations for contextual semantics used in Experiment 3.  
- **`ontology_dump.txt`** — Simplified ontology used within the *Ontology-Based SPARQL Repair (OBSR)* module (Experiment 4).  
- **`NL2SPARQL_FT_DATA_1600.json`** — Fine-tuning dataset containing 1,600 NL–SPARQL pairs (used to finetune the model in Experiment 2).

---

## How to Reproduce

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/nl2sparql-rag.git
   cd nl2sparql-rag

---

## Author

**Vicky Vicky**  
*M.Sc. in Artificial Intelligence*  
[Lehrstuhl für Mustererkennung (Informatik 5)](https://www5.cs.fau.de/)  
Department Informatik,  
**Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**

---

## Supervisors

- **Dr. Tri-Thien Nguyen, M.Sc.**  
- **Dr. rer. medic. Soroosh Tayebi Arasteh, M.Sc.**  
- **Dr.-Ing. Andreas Maier**  
- **M.A. Detlef Grittner (CTO)**  
- **Dipl. Inf. Peter Feltens (CEO)**  
