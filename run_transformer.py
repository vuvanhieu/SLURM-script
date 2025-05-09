from sentence_transformers import SentenceTransformer

print("✅ [Transformer] Import successful")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(["This is a test sentence."], show_progress_bar=False)
    print("✅ [Transformer] Embedding generated. Shape:", embedding.shape)
except Exception as e:
    print("❌ [Transformer] Error:", e)
