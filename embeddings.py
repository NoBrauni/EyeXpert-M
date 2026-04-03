import os
import pickle
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel

# -------------------------------
# Initialize encoder/tokenizer
# -------------------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# -------------------------------
# Helpers
# -------------------------------
def normalize_sentence(s):
    return " ".join(s.strip().lower().split())

def get_word_spans(sentence):
    words = []
    spans = []
    start = 0

    for word in sentence.split():
        start = sentence.find(word, start)
        end = start + len(word)
        words.append(word)
        spans.append((start, end))
        start = end

    return words, spans

# -------------------------------
# Main embedding function
# -------------------------------
def generate_universal_sentence_embeddings(
    pickle_dir,
    cache_path="full_sentence_embeddings.pkl",
    batch_size=16,
    device=None,
    max_sentences=None,     # limit number of sentences for testing
    restrict_file=None      # process only one dataset file
):
    """
    Builds a universal embedding cache:
    {
        normalized_sentence: {
            "embeddings": Tensor[num_words, hidden_dim] (on device),
            "words": [word1, word2, ...]
        }
    }
    """

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move encoder to device
    encoder.to(device)

    # Load or initialize cache
    if os.path.exists(cache_path):
        print(f"✅ Loading existing embedding cache from {cache_path}")
        with open(cache_path, "rb") as f:
            sentence_cache = pickle.load(f)
    else:
        sentence_cache = {}

    # Iterate over dataset files
    for fname in os.listdir(pickle_dir):
        if not fname.endswith(".pkl"):
            continue

        # Restrict to a single file if specified
        if restrict_file and fname != restrict_file:
            continue

        path = os.path.join(pickle_dir, fname)
        with open(path, "rb") as f:
            samples = pickle.load(f)

        # Collect sentences needing embeddings
        sentences_to_compute = list({
            normalize_sentence(s["sentence"])
            for s in samples
            if s.get("sentence") is not None
               and normalize_sentence(s["sentence"]) not in sentence_cache
        })

        # Limit for testing
        if max_sentences:
            sentences_to_compute = sentences_to_compute[:max_sentences]

        print(f"{fname}: {len(sentences_to_compute)} sentences to process")

        # Process in batches
        for start in range(0, len(sentences_to_compute), batch_size):
            batch_sentences = sentences_to_compute[start:start + batch_size]

            # Tokenize
            encoding = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True
            )

            offsets = encoding.pop("offset_mapping")
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Encode
            with torch.no_grad():
                outputs = encoder(**encoding)

            hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

            # Align tokens → words
            for i, sentence in enumerate(batch_sentences):
                token_embeds = hidden[i]                # [seq_len, hidden_dim]
                token_offsets = offsets[i].tolist()     # [(start, end), ...]

                words, word_spans = get_word_spans(sentence)
                word_vectors = []

                for w_start, w_end in word_spans:
                    subword_vectors = []

                    for tok_i, (t_start, t_end) in enumerate(token_offsets):
                        # skip special tokens (<s>, </s>)
                        if t_start == t_end:
                            continue

                        # check overlap
                        if not (t_end <= w_start or t_start >= w_end):
                            subword_vectors.append(token_embeds[tok_i])

                    if subword_vectors:
                        word_vectors.append(torch.stack(subword_vectors).mean(dim=0))
                    else:
                        # safe fallback
                        word_vectors.append(torch.zeros_like(token_embeds[0]))

                word_tensor = torch.stack(word_vectors).to(device)

                # Sanity check
                assert len(words) == word_tensor.size(0), \
                    f"Mismatch: {len(words)} words vs {word_tensor.size(0)} embeddings"

                # Store normalized sentence as key
                sentence_cache[normalize_sentence(sentence)] = {
                    "embeddings": word_tensor,
                    "words": words
                }

            print(f"Processed {start + len(batch_sentences)}/{len(sentences_to_compute)}")

    # Save cache (always to CPU for portability)
    cpu_cache = {k: {"embeddings": v["embeddings"].cpu(), "words": v["words"]}
                 for k, v in sentence_cache.items()}

    with open(cache_path, "wb") as f:
        pickle.dump(cpu_cache, f)

    print(f"✅ Saved embedding cache at {cache_path} on CPU")

# -------------------------------
# Example usage (SAFE TEST)
# -------------------------------
if __name__ == "__main__":
    generate_universal_sentence_embeddings(
        pickle_dir="datasets",
        cache_path="test_embeddings.pkl",
        batch_size=4,                   # small for CPU
        device=None,                     # auto detect GPU/CPU
        restrict_file="meco_dataset_en.pkl",  # only one language
        max_sentences=50                # small subset
    )