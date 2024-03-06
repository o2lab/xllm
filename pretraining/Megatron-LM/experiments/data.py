from datasets import load_dataset

train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train', cache_dir="/scratch/user/siweicui/LLM/huggingface/")
train_data.to_json("codeparrot_data.json", lines=True)  