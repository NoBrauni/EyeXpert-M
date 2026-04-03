import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel

from model_definition import EyeExpertM, LANG_TO_EXPERT

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "checkpoints/eyeexpert_epoch1.pt"
DURATION_STATS = "duration_stats.npy"

beam_size = 5
max_steps = 30
temperature = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------------
# Load duration stats
# -------------------------------
dur_mean, dur_std = np.load(DURATION_STATS)
print(f"Duration stats loaded: mean={dur_mean:.2f} ms, std={dur_std:.2f} ms")

# -------------------------------
# Load encoder
# -------------------------------
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# -------------------------------
# Load trained model
# -------------------------------
model = EyeExpertM(
    hidden_dim=256,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=53,  # make sure this matches your training
    window_size=4,
    n_layers=2,
    attention_type="dot",
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

# Attempt to load state_dict directly
try:
    model.load_state_dict(checkpoint)
except RuntimeError as e:
    print("Warning: Could not load state_dict cleanly:", e)
model.eval()
print("Model loaded")

# -------------------------------
# Word embeddings for a new sentence
# -------------------------------
def compute_word_embeddings(sentence):
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=False,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"][0]
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

    token_emb = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # Split sentence into words (same assumption as training)
    words = sentence.split()

    word_vectors = []
    token_used = set()

    for word in words:
        word_clean = word.strip().lower()
        subword_vectors = []

        for tok_i, (start, end) in enumerate(offsets):
            token_text = sentence[start:end].strip().lower()

            # More robust match than exact equality
            if token_text == word_clean:
                subword_vectors.append(token_emb[tok_i])
                token_used.add(tok_i)

        if len(subword_vectors) > 0:
            word_vec = torch.stack(subword_vectors).mean(dim=0)
        else:
            # fallback: use first token embedding (same as training fallback)
            word_vec = token_emb[0]

        word_vectors.append(word_vec)

    return torch.stack(word_vectors)  # [num_words, hidden_dim]

# -------------------------------
# Beam search scanpath generation
# -------------------------------
import torch
import numpy as np

def top_p_filtering(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    cutoff = cumulative_probs > top_p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False

    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    return sorted_probs, sorted_indices


def generate_scanpath(sentence, lang="en", top_p=0.9, temperature=0.8):
    expert_id = LANG_TO_EXPERT.get(lang.lower(), 0)

    word_embeddings = compute_word_embeddings(sentence).unsqueeze(0).to(device)
    num_words = word_embeddings.size(1)

    fix_seq = [0]
    dur_seq = []

    for step in range(max_steps):
        fix_tensor = torch.tensor([fix_seq], dtype=torch.long).to(device)
        inputs = word_embeddings[:, fix_seq]
        lengths = [len(fix_seq)]

        with torch.no_grad():
            logits, dur_pred = model(inputs, fix_tensor, word_embeddings, lengths, expert_id)

        # Take last timestep
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)

        # Apply top-p filtering
        sorted_probs, sorted_indices = top_p_filtering(probs, top_p=top_p)

        # Sample from filtered distribution
        next_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
        next_fix = sorted_indices[next_idx_in_sorted].item()

        # Prevent invalid indices
        next_fix = min(next_fix, num_words - 1)

        # Duration (same as before)
        dur_norm = dur_pred[0, -1].item()
        dur_ms = dur_norm * dur_std + dur_mean

        fix_seq.append(next_fix)
        dur_seq.append(dur_ms)

        # Stop if we've covered all words
        if len(set(fix_seq)) >= num_words:
            break

        if len(fix_seq) >= max_steps:
            break

    return fix_seq, dur_seq

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog"
    fixations, durations = generate_scanpath(sentence, lang="en")
    words = sentence.split()

    print("\nSentence:")
    print(sentence)
    print("\nScanpath:")
    for i, (fix, dur) in enumerate(zip(fixations, durations)):
        word = words[fix] if fix < len(words) else "?"
        print(f"{i:02d}: {word}  ({dur:.1f} ms)")