from safetensors.torch import load_file, save_file

# Path to your adapter safetensors
adapter_path = "./gemma_lora_adap_edit/adapter_model.safetensors"

# Load the state dict
state_dict = load_file(adapter_path)

# Keys causing issues
bad_keys = [
    "base_model.model.model.embed_tokens.weight",
    "base_model.model.lm_head.weight"
]

# Delete the bad keys
for key in bad_keys:
    if key in state_dict:
        print(f"Removing {key} from adapter")
        del state_dict[key]

# Save cleaned adapter
save_file(state_dict, "./adapter_model.safetensors")