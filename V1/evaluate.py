import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from model_definition import EyeExpertM, LANG_TO_EXPERT

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ Load the trained model
# -------------------------------
model = EyeExpertM(hidden_dim=256, encoder_dim=768, n_experts=5, max_seq_len=200).to(device)
checkpoint_path = "checkpoints/eyeexpert_epoch1.pt"  # change if needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# -------------------------------
# 3️⃣ Tokenizer and encoder
# -------------------------------
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# -------------------------------
# 4️⃣ Prepare the new sentence
# -------------------------------
new_sentence = "This is a test sentence for eye-tracking prediction."
words = new_sentence.strip().split()

# Compute embeddings
encoding = tokenizer([new_sentence], return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
with torch.no_grad():
    outputs = encoder(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
hidden_states = outputs.last_hidden_state

word_embeddings = []
offsets = encoding["offset_mapping"][0]
for word in words:
    vecs = []
    for j, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        token_text = new_sentence[start:end]
        if token_text.strip() == word.strip():
            vecs.append(hidden_states[0, j])
    if vecs:
        word_embeddings.append(torch.stack(vecs).mean(dim=0))
    else:
        word_embeddings.append(torch.zeros(hidden_states.size(-1)))
word_embeddings = torch.stack(word_embeddings).unsqueeze(0).to(device)  # [1, seq_len, encoder_dim]

# -------------------------------
# 5️⃣ Dummy fix_seq for scanpath (0..seq_len-1)
# -------------------------------
seq_len = word_embeddings.size(1)
fix_seq = torch.arange(seq_len).unsqueeze(0).to(device)

# -------------------------------
# 6️⃣ Choose expert (for demo, English = expert 0)
# -------------------------------
expert_id = LANG_TO_EXPERT.get("en", 0)

# -------------------------------
# 7️⃣ Run model
# -------------------------------
with torch.no_grad():
    logits, dur_pred = model(word_embeddings, fix_seq, word_embeddings, [seq_len], expert_id)

pred_fix_indices = logits.argmax(dim=-1)[0].tolist()
pred_durations = dur_pred[0].tolist()

# -------------------------------
# 8️⃣ Print results
# -------------------------------
print("Words:", words)
print("Predicted next-fixation indices:", pred_fix_indices)
print("Predicted durations (s):", pred_durations)