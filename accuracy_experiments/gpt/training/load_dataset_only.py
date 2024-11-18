print("Loading modules...")
import src.datamodules.language_modeling_hf as l
print("Initializing datamodule...")
a = l.LMDataModule(dataset_name="openwebtext", tokenizer_name="gpt2", cache_dir="data/openwebtext/cache", max_length=1024, val_ratio=0.0005, val_split_seed=2357, add_eos=True, batch_size=8, batch_size_eval=16, shuffle=True, pin_memory=True)
print("Preparing data...")
a.prepare_data()
print("Setting up...")
a.setup()
