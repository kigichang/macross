from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as nn

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
print(model)
# prepare input
text = "Hello I'm a <mask> model."
inputs = tokenizer(text, return_tensors='pt')

# forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits


# Find the `[MASK]` token index
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# Extract logits for the `[MASK]` position
mask_token_logits = logits[0, mask_token_index, :]

# Apply softmax to get probabilities
probabilities = nn.softmax(mask_token_logits, dim=1)

# Get the top 5 predicted tokens and their probabilities
top_5 = torch.topk(probabilities, 5, dim=1)
top_5_tokens = top_5.indices[0].tolist()  # Token IDs
top_5_probs = top_5.values[0].tolist()    # Probabilities

# Decode the tokens and print results
predicted_words_with_probs = [
    (tokenizer.decode([token]).strip(), prob)
    for token, prob in zip(top_5_tokens, top_5_probs)
]

for word, prob in predicted_words_with_probs:
    print(f"{word}: {prob:.3f}")