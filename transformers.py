import torch, transformers
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
assert hasattr(torch, 'frombuffer'), "Torch missing required attributes!"