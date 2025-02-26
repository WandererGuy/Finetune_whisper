from datasets import load_dataset

ds = load_dataset("jlvdoorn/atco2-asr-atcosim", cache_dir="./manh_data")


# from datasets import DatasetDict, Dataset

# common_voice = DatasetDict()

# # Download and prepare the dataset locally (without loading it immediately)
# dataset = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
# dataset.download_and_prepare(download_dir="./data")  # Downloads and stores dataset in "./data" folder
