import os
os.environ["http_proxy"] = "http://proxy:80"
os.environ["https_proxy"] = "http://proxy:80"
os.environ["HTTP_proxy"] = "http://proxy:80"
os.environ["HTTPS_proxy"] = "http://proxy:80"

os.environ["HF_HOME"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"

from huggingface_hub import login
login(token="hf_CFcHTG********************")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "allenai/OLMo-2-1124-13B-Instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#model_name = "google/gemma-2-9b-it"

#try this with the unsloth model as well

# Load model with automatic device placement
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #cache_dir=path_name,
    torch_dtype=torch.bfloat16,  #Use bfloat16 for memory efficiency
    device_map="auto"  #multiple GPUs
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Print device map (to confirm layers are split across GPUs)
#print(model.hf_device_map)
#print(type(tokenizer))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class DocumentProcessor:
    def __init__(self, ontology_path):
        self.ontology_path = ontology_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    def process_ontology(self):
        with open(self.ontology_path, 'r') as file:
            ontology_text = file.read()
        chunks = self.text_splitter.split_text(ontology_text)
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

    def retrieve_relevant_chunks(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)


class QueryGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_sparql_query(self, nl_query, context):
        prompt = f"""You are given two inputs:

1. **Ontology context**: A list of RDF/OWL property and class definitions. All property names are case-sensitive.
2. **Natural language query**: A question about the data described by the ontology.

Your task is to generate a **complete and correct SPARQL query** that answers the question, using only the given ontology terms.

Follow these strict rules:

- Exclusively Generate SPARQL query. Do not include any explanations or extra text.
- Make use of rdfs:comment that will help you to build semantically correct Query.
- Do not use the property "dcm:hasInformationEntity". It is incorrect for all cases.
- Always include full SPARQL syntax with correct PREFIX declarations at the top.
- All property and class names must exactly match the terms in the ontology (case-sensitive).


Example SPARQL query format (for reference only):

PREFIX dcm: <http://semantic-dicom.org/dcm#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?patientName ?studyUID
WHERE {{
    ?patient rdf:type dcm:Patient .
    ?patient dcm:PatientName ?patientName .

    ?study rdf:type dcm:Study .
    ?study dcm:isStudyOf ?patient .

    ?study dcm:StudyInstanceUID ?studyUID .
}}

---

Ontology context:
{context}

Question:
{nl_query}

SPARQL query:
"""
        #max_length = self.model.config.max_position_embeddings - 100 , truncation=True, max_length=max_length
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sparql_query = response.split("SPARQL query:")[-1].strip()
        return sparql_query

class RAGSystem:
    def __init__(self, model, tokenizer, ontology_path):
        self.document_processor = DocumentProcessor(ontology_path)
        self.document_processor.process_ontology()
        self.query_generator = QueryGenerator(model, tokenizer)

    def generate_sparql_from_nl(self, nl_query):
        relevant_chunks = self.document_processor.retrieve_relevant_chunks(nl_query)
        context = "\n".join([chunk.page_content for chunk in relevant_chunks])
        sparql_query = self.query_generator.generate_sparql_query(nl_query, context)
        return sparql_query

#it throws an error "kernel not found" so here is the solution:
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

import json
import os
import pandas as pd
def gsheet_to_json():
    test_data = "https://docs.google.com/spreadsheets/d/1BL1TGPg4m23aP_D8KzF5Ura8ZDvfe4uDUGDNEfGflB0/edit?gid=1263403686#gid=1263403686"
    s_id = test_data.split("/d/")[1].split("/")[0]
    csv_export_url = f"https://docs.google.com/spreadsheets/d/{s_id}/export?format=csv"
    df = pd.read_csv(csv_export_url, dtype=str)
    pd.set_option('display.max_colwidth', None)

    test_data = '/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/Test_Data.json'

    #OPEN AN EMPTY JSON FILE
    with open(test_data, 'w') as file:
        json.dump([], file)

    for row in range(df.shape[0]):
        data = {
            'Natural_Query': df.iloc[row, 0]
        }
        with open(test_data, 'r') as file:
            existing_data = json.load(file)

            existing_data.append(data)

            with open(test_data, 'w') as file:
                json.dump(existing_data, file, indent=4)

gsheet_to_json()

ontology_path = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/onto_ver_7.owl"
rag_system = RAGSystem(model, tokenizer, ontology_path)

#get all the queries in "data" variable
with open("/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/Test_Data.json", "r") as file:
    data = json.load(file)

sparql_output = '/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/sparql_mistral7b_exp1.txt'
#create a new file if file exist it will delete the existing data
f = open(sparql_output, "w")

n = 0

for item in data:
    n+=1
    nl_query = (f"\"{item['Natural_Query']}\"")
    sparql_query = rag_system.generate_sparql_from_nl(nl_query)
    f = open(sparql_output, "a")
    f.write(f"Question{n}: " + nl_query+"\n\n")
    f.write(sparql_query+"\n\n\n\n\n\n" )
    f.close()



import torch

output_file = "gpu_info.txt"

with open(output_file, "w") as f:
    f.write(f"Torch Version: {torch.__version__}\n")
    f.write(f"Number of GPUs: {torch.cuda.device_count()}\n")

    for i in range(torch.cuda.device_count()):
        f.write(f"\nGPU {i}: {torch.cuda.get_device_name(i)}\n")
        f.write(f"Memory Usage: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB\n")

print(f"GPU information saved to {output_file}")
