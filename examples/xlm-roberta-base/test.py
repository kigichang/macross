from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base",  use_safetensors=False)
print("model")
print(model)

model.eval()
output_stat_dict = model.state_dict()
for key in output_stat_dict:
    if "beta" in key or "gamma" in key:
        print("warning: old format name:", key)
torch.save(output_stat_dict, "fix-xml-roberta-base.pth")

# prepare input
text = ["Hello I'm a <mask> model."]
encoded_input = tokenizer(text, return_tensors='pt')
print("encoded_input")
print(encoded_input)

# forward pass
output = model(**encoded_input)

mask_token_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
logits = output.logits
probs = F.softmax(logits[0, mask_token_index, :], dim=1)
top_5 = probs.topk(5, dim=1)

for i in range(5):
    token_id = top_5.indices[0, i].item()
    score = top_5.values[0, i].item()
    token_str = tokenizer.decode([token_id])
    print(token_str, score)