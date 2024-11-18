# This script is tuned to fit the models in A100-40GB GPUs


# Reproducing the SLoPe results

# Training

python -m performance_experiments.memory --model "facebook/opt-2.7b" --num_hidden_layers 12
python -m performance_experiments.memory --model "facebook/opt-6.7b" --num_hidden_layers 12
python -m performance_experiments.memory --model "facebook/opt-13b" --num_hidden_layers 6
python -m performance_experiments.memory --model "facebook/opt-30b" --num_hidden_layers 3
python -m performance_experiments.memory --model "facebook/opt-66b" --num_hidden_layers 2

python -m performance_experiments.memory --model "meta-llama/Meta-Llama-3-8B" --num_hidden_layers 4
python -m performance_experiments.memory --model "mistralai/Mistral-7B-v0.3" --num_hidden_layers 2

# Inference without LoRA

python -m performance_experiments.memory --model "facebook/opt-2.7b" --num_hidden_layers 12 --inference_only
python -m performance_experiments.memory --model "facebook/opt-6.7b" --num_hidden_layers 12 --inference_only
python -m performance_experiments.memory --model "facebook/opt-13b" --num_hidden_layers 12 --inference_only
python -m performance_experiments.memory --model "facebook/opt-30b" --num_hidden_layers 6 --inference_only
python -m performance_experiments.memory --model "facebook/opt-66b" --num_hidden_layers 3 --inference_only

python -m performance_experiments.memory --model "meta-llama/Meta-Llama-3-8B" --num_hidden_layers 4 --inference_only
python -m performance_experiments.memory --model "mistralai/Mistral-7B-v0.3" --num_hidden_layers 2 --inference_only

# Inference with LoRA

for RANK in 0.015625 0.0625
do
    python -m performance_experiments.memory --model "facebook/opt-2.7b" --num_hidden_layers 12 --inference_only \
        --add_lora --lora_rank $RANK
    python -m performance_experiments.memory --model "facebook/opt-6.7b" --num_hidden_layers 12 --inference_only \
        --add_lora --lora_rank $RANK
    python -m performance_experiments.memory --model "facebook/opt-13b" --num_hidden_layers 12 --inference_only \
        --add_lora --lora_rank $RANK
    python -m performance_experiments.memory --model "facebook/opt-30b" --num_hidden_layers 6 --inference_only \
        --add_lora --lora_rank $RANK
    python -m performance_experiments.memory --model "facebook/opt-66b" --num_hidden_layers 3 --inference_only \
        --add_lora --lora_rank $RANK

    python -m performance_experiments.memory --model "meta-llama/Meta-Llama-3-8B" --num_hidden_layers 4 \
        --inference_only --add_lora --lora_rank $RANK
    python -m performance_experiments.memory --model "mistralai/Mistral-7B-v0.3" --num_hidden_layers 2 \
        --inference_only --add_lora --lora_rank $RANK
done


# Reproducing the results in Accelerating Transformer Pre-Training with 2:4 Sparsity
# https://arxiv.org/abs/2404.01847

python -m performance_experiments.memory --model "facebook/opt-2.7b" --num_hidden_layers 12 --dynamic
python -m performance_experiments.memory --model "facebook/opt-6.7b" --num_hidden_layers 12 --dynamic
python -m performance_experiments.memory --model "facebook/opt-13b" --num_hidden_layers 3 --dynamic
python -m performance_experiments.memory --model "facebook/opt-30b" --num_hidden_layers 3 --dynamic
python -m performance_experiments.memory --model "facebook/opt-66b" --num_hidden_layers 2 --dynamic

python -m performance_experiments.memory --model "meta-llama/Meta-Llama-3-8B" --num_hidden_layers 4 --dynamic
python -m performance_experiments.memory --model "mistralai/Mistral-7B-v0.3" --num_hidden_layers 2 --dynamic
