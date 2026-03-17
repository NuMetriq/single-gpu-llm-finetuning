from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/no_robots")

print(dataset)
print("\nAvailable splits:", dataset.keys())
print("\nFirst training example:")
print(dataset["train"][0])
