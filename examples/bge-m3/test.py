from transformers import AutoTokenizer, AutoModel
import torch

def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]

# 模型名稱
MODEL_NAME = "BAAI/bge-m3"

# 加載模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# print("tokenizer")
# print(tokenizer)

model = AutoModel.from_pretrained(MODEL_NAME)
print("model")
print(model)

# 文本列表
#texts = ["這是一個文本", "這是另一個文本"]
texts = ["What is BGE M3?", "Defination of BM25"]

# 文本處理
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
# print("inputs")
# print(inputs)

# 獲取嵌入
with torch.no_grad():
    model_output = model(**inputs)
    print("model_output")
    print(model_output[0])
    # print(outputs)
    # print("last_hidden_state")
    # print(outputs.last_hidden_state)
    # sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
    # print("sentence_embeddings - mean")
    # print(sentence_embeddings)

# 嵌入歸一化
# normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
# print("Normalized Embeddings:")
# print(normalized_embeddings)
# 計算相似度
# similarity = torch.nn.functional.cosine_similarity(normalized_embeddings[0].unsqueeze(0), normalized_embeddings[1].unsqueeze(0))
# print("Cosine Similarity:", similarity.item())

# sentence_embeddings = cls_pooling(model_output, inputs['attention_mask'])
# print("Sentence embeddings:")
# print(sentence_embeddings.shape)
# print(sentence_embeddings)