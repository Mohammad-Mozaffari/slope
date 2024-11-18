from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


hf_token = "YOUR_HF_TOKEN"


for model_name in [
                    "google-bert/bert-large-uncased",
                    "openai-community/gpt2",
                    "openai-community/gpt2-medium",
                    "openai-community/gpt2-large",
                    "openai-community/gpt2-xl", 
                    "facebook/opt-125m",
                    "facebook/opt-350m",
                    "facebook/opt-1.3b",
                    "facebook/opt-2.7b",
                    "facebook/opt-6.7b",
                    "facebook/opt-13b",
                    "facebook/opt-30b",
                    "facebook/opt-66b",
                    "meta-llama/Meta-Llama-3-8B",
                    "mistralai/Mistral-7B-v0.3",
                   ]:
    config = AutoConfig.from_pretrained(model_name, cache_dir="cache", token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache", token=hf_token)
