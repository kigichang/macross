from transformers import AutoTokenizer, AutoModel
import torch

# 模型名稱
MODEL_NAME = "BAAI/bge-m3"

# 加載模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
print("model")
print(model)

# 文本列表
texts = ["這是一個文本", "這是另一個文本"]

# 文本處理
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 獲取嵌入
with torch.no_grad():
    outputs = model(**inputs)
    #print(outputs)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

# 嵌入歸一化
normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

# 計算相似度
similarity = torch.nn.functional.cosine_similarity(normalized_embeddings[0].unsqueeze(0), normalized_embeddings[1].unsqueeze(0))
print("Cosine Similarity:", similarity.item())
