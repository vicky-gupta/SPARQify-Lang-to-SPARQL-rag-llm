***Thesis Topic: LLM-Centric Framework for Ontology-Driven SPARQL Query Generation in RAG for DICOM Databases***

*Author:*

Vicky Vicky
M.Sc. in Artificial Intelligence
Lehrstuhl für Mustererkennung (Informatik 5)
Department Informatik
Friedrich-Alexander-Universität Erlangen-Nürnberg.

*Supervisors:*

Dr. Tri-Thien Nguyen,
M.Sc., Dr. rer. medic. Soroosh Tayebi Arasteh,
M.Sc., Dr.-Ing. Andreas Maier,
M.Sc., M.A. Detlef Grittner (CTO),
Dipl. Inf. Peter Feltens (CEO)

**BODY OF THE PAGE:**

This folder contains all the assets required to reproduce the results of this thesis. It contains 5 folders that include all Python scripts, SLURM scripts, and datasets used in the implementation of the thesis.

Note: While reproducing, you should recheck all the paths in the scripts because all of these experiments were originally conducted on an HPC (High Performance Computing) system. During the transfer of the assets to a local secure folder, the original file structure was lost.

**File & Folders:**

*requirements.txt:* All the packages and modules needed to reproduce the results.

*Experiments_Scripts:* All four experiment codes, including the fine-tuning code for a multi-GPU setup.

*Datasets:*

- It contains the dataset used for fine-tuning and the ontology data used in all the experiments.
- *onto_ver_7.owl:* The 7th version of the ontology, used as a RAG document in Experiment 1.
- *onto_ver_7_semantics.owl:* A version of the ontology that includes `rdfs:comment` annotations to add contextual clues.
- *ontology_dump.txt:* A simplified version of the ontology used in Experiment 4 within the OBSR (Ontology-Based SPARQL Repair) Module.
- *NL2SPARQL_FT_DATA_1600.json:* A fine-tuning dataset of 1600 entries containing NL2SPARQL pairs.

*HPC_Scripts:* All the SLURM scripts used to submit jobs to the HPC.

*Fine_Tuning_Weights:*

- Contains all the training weights required for Experiment 2.
- Includes a script (*remove_bad_keys.py*) used to correct the fine-tuning weights by removing problematic keys. This step is necessary for the weights to work properly in Experiment 2.

*Sparql_Results:* All SPARQL result files in `.txt` format.




