#FINETUNED_MODEL + RAG + Ontology_Semantics
import os
os.environ["http_proxy"] = "http://proxy:80"
os.environ["https_proxy"] = "http://proxy:80"
os.environ["HTTP_proxy"] = "http://proxy:80"
os.environ["HTTPS_proxy"] = "http://proxy:80"

os.environ["HF_HOME"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"

from huggingface_hub import login
login(token="hf_CFcHT*****************")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
#load model wieghts
adapter_path = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/mistral_lora_adap_edit"
model = PeftModel.from_pretrained(model, adapter_path)

# Optional: Merge weights for faster inference (removes the adapter architecture overhead)
# model = model.merge_and_unload()


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
        self.INSTRUCTION_START = "<instruction>"
        self.INSTRUCTION_END = "</instruction>"
        self.QUESTION_START = "<question>"
        self.QUESTION_END = "</question>"
        self.SPARQL_START = "<sparql>"
        self.SPARQL_END = "</sparql>"

    def generate_sparql_query(self, nl_query, context):
        # Format exactly as in training
        instruction = f"{self.INSTRUCTION_START}Generate a SPARQL query from the given natural language question using the provided ontology context. Use only terms from the ontology.{self.INSTRUCTION_END}"
        question = f"{self.QUESTION_START}Ontology context:\n{context}\n\nQuestion: {nl_query}{self.QUESTION_END}"

        # Start the SPARQL generation - this format must match training
        prompt = f"{instruction}\n{question}\n{self.SPARQL_START}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate with appropriate parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1
        )

        # Decode with special tokens preserved
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract the SPARQL query between the tags
        try:
            if self.SPARQL_END in response:
                sparql_query = response.split(self.SPARQL_START)[1].split(self.SPARQL_END)[0].strip()
            else:
                # Fallback if model didn't generate end tag
                sparql_query = response.split(self.SPARQL_START)[1].strip()
        except IndexError:
            # Handle case where the model output doesn't match expected format
            return "Error: Could not extract SPARQL query from model output"

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

ontology_path = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/onto_ver_7_semantics.owl"
rag_system = RAGSystem(model, tokenizer, ontology_path)

#get all the queries in "data" variable
with open("/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/Test_Data.json", "r") as file:
    data = json.load(file)

sparql_output = '/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/sparql_mistral7b_exp3.txt'
#create a new file if file exist it will delete the existing data
f = open(sparql_output, "w")

n = 0


######______commenting for short time

for item in data:
    n+=1
    nl_query = (f"\"{item['Natural_Query']}\"")
    sparql_query = rag_system.generate_sparql_from_nl(nl_query)
    f = open(sparql_output, "a")
    f.write(f"Question{n}: " + nl_query+"\n\n")
    f.write(sparql_query+"\n\n\n\n\n\n" )
    f.close()
    #print(nl_query)
