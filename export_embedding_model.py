# export_embedding_model.py
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
save_path = "models/all-MiniLM-L6-v2"   # 存到项目内的 models 目录

print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name)

print(f"Saving model to: {save_path}")
model.save(save_path)

print("Done.")
