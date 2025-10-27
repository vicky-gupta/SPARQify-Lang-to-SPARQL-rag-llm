#UPDATED LORA FINE TUNING APPROACH + MULTI GPU + SAVING THE MODEL
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

os.environ["http_proxy"] = "http://proxy:80"
os.environ["https_proxy"] = "http://proxy:80"
os.environ["HTTP_proxy"] = "http://proxy:80"
os.environ["HTTPS_proxy"] = "http://proxy:80"

os.environ["HF_HOME"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/.cache/huggingface"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

from huggingface_hub import login

login(token="hf_CFcHTG******************", add_to_git_credential=False)


# ---------------------------
# 1. Load Model and Tokenizer
# ---------------------------
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#model_name = "allenai/OLMo-2-1124-13B-Instruct"
model_name = "google/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # Use bfloat16 for memory efficiency
    #device_map="auto"             # Automatically distribute across available GPUs
)
model.gradient_checkpointing_enable() #Added Part
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 512 # tokenizer.model_max_length
# Define delimiter tags for consistent use throughout the code
INSTRUCTION_START = "<instruction>"
INSTRUCTION_END = "</instruction>"
QUESTION_START = "<question>"
QUESTION_END = "</question>"
SPARQL_START = "<sparql>"
SPARQL_END = "</sparql>"

# Add these as special tokens to the tokenizer
special_tokens_dict = {
    "additional_special_tokens": [
        INSTRUCTION_START, INSTRUCTION_END,
        QUESTION_START, QUESTION_END,
        SPARQL_START, SPARQL_END
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)

# Resize model embeddings to account for the new tokens
model.resize_token_embeddings(len(tokenizer))

# ---------------------------
# 2. Load and Preprocess Dataset
# ---------------------------

dataset = load_dataset("json", data_files="NL2SPARQL_FT_DATA_1600.json")["train"]
train_val_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]

def format_example(example):

    # Format prompt with clear XML-like tags
    instruction = f"{INSTRUCTION_START}{example['instruction']}{INSTRUCTION_END}"
    question = f"{QUESTION_START}{example['question']}{QUESTION_END}"

    # This is what we want the model to see before generating
    prompt = f"{instruction}\n{question}\n{SPARQL_START}"

    # This is what we want the model to generate
    completion = f"{example['sparql']}{SPARQL_END}"

    # Tokenize prompt and completion separately
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    completion_tokens = tokenizer(completion, add_special_tokens=False)

    # Combine them for the full input
    input_ids = prompt_tokens["input_ids"] + completion_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + completion_tokens["attention_mask"]

    # Critical: set labels to -100 for prompt (ignored in loss calculation)
    # Only the completion part will contribute to the loss
    labels = [-100] * len(prompt_tokens["input_ids"]) + completion_tokens["input_ids"]

    # Make sure everything fits within max_length
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        # Ensure we don't cut in the middle of the completion if possible
        prompt_length = len(prompt_tokens["input_ids"])
        if prompt_length < MAX_LENGTH:
            labels = labels[:MAX_LENGTH]
        else:
            # If prompt is too long, we can't use this example effectively
            labels = [-100] * MAX_LENGTH
    else:
        # Pad to max length if needed
        padding_length = MAX_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Apply the formatting to the dataset
tokenized_train_dataset = train_dataset.map(format_example, batched=False)
tokenized_val_dataset = val_dataset.map(format_example, batched=False)

# ---------------------------
# 3. Configure LoRA for Fine-Tuning
# ---------------------------
lora_config = LoraConfig(
    r=4, #8                    # Rank of the low-rank matrices; tweak as needed for capacity
    lora_alpha=8, #32        # Scaling factor for updates
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Targeting typical projection layers; adjust based on the model architecture
    lora_dropout=0.1,
    bias="none"             # Do not update biases in LoRA
)

# Wrap the base model with the PEFT model (using LoRA)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # Log the number of trainable parameters

# ---------------------------
# 4. Setup Training
# ---------------------------
# training_args = TrainingArguments(
#     output_dir="./lora_finetuned_model",
#     per_device_train_batch_size=2,  # Adjust based on available memory and GPU count
#     num_train_epochs=3,             # Set to an appropriate number of epochs
#     learning_rate=3e-4,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,                      # Enable mixed precision training for efficiency
# )

####DEEPSPEED_ARGUMENTS#####

training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=2e-5,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=False,
    fp16=False,
    bf16=True,
    deepspeed="/home/woody/iwi5/iwi5276h/MT_thesis_SOHARD/deepspeed_config.json",
    report_to="none"
)

# For causal language modeling, we use a data collator that does not perform masked LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,  # Add this line
    data_collator=data_collator
)

# ---------------------------
# 5. Fine-Tuning
# ---------------------------
trainer.train()

# Save only the LoRA adapter weights (which represent the fine-tuned parameters)
peft_model.save_pretrained("./gemma_lora_adap_weights")

# Optionally, if you want a standalone (merged) model for inference,
# you can merge the LoRA weights into the base model as follows:
# merged_model = peft_model.merge_and_unload()
# merged_model.save_pretrained("./lora_merged_model")
