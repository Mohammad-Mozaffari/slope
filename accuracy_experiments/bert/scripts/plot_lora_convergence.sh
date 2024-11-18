cd ../


CMD="python utils/plot.py --file_list results/checkpoint_similarity.csv --lora_convergence"
echo $CMD
$CMD
