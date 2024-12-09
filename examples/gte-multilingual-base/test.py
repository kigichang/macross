# Requires transformers>=4.36.0

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "北京",
    "快排算法介绍"
]

model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print("tokenizer")
print(tokenizer)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
print("model")
print(model)

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)

dimension=768 # The output dimension of the output embedding, should be in [128, 768]
embeddings = outputs.last_hidden_state[:, 0][:dimension]

embeddings = F.normalize(embeddings, p=2, dim=1)
print("embeddings")
print(embeddings)
# tensor([[-0.0736,  0.0594, -0.0844,  ..., -0.0152, -0.0313, -0.0310],
#         [-0.0586,  0.0612, -0.0755,  ...,  0.0383, -0.0133, -0.0218],
#         [-0.1061,  0.0809, -0.0899,  ...,  0.0040,  0.0358, -0.0461],
#         [-0.0284,  0.0705, -0.1145,  ...,  0.0454,  0.0360, -0.0614]],
#        grad_fn=<DivBackward0>)

scores = (embeddings[:1] @ embeddings[1:].T) * 100
print("scores")
print(scores.tolist())

# [[0.3016996383666992, 0.7503870129585266, 0.3203084468841553]]
