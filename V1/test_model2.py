import pickle
import torch

cache_path = "all_embeddings_cache.pkl"
with open(cache_path, "rb") as f:
    embedding_cache = pickle.load(f)

# Type of the cache
print("Cache type:", type(embedding_cache))

# Show first few keys
first_keys = list(embedding_cache.keys())[:5]
print("First keys:", first_keys)

# Check the type and shape of the first embedding
first_val = embedding_cache[first_keys[0]]
print("First value type:", type(first_val))
if isinstance(first_val, torch.Tensor):
    print("Shape of first embedding:", first_val.shape)