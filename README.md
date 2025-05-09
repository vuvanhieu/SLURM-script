# 🔧 LLM + GNN SLURM Pipeline Template

This repository provides a fully working SLURM-compatible pipeline for multitask models using MLP, GNN, and Transformers. It is designed to run on systems with CUDA 11.2 and PyTorch 1.9.1 using precompiled PyG wheels from [https://data.pyg.org](https://data.pyg.org).

---

## 📦 Components

- ✅ GPU availability check (NVIDIA-SMI, PyTorch, TensorFlow)
- ✅ SentenceTransformer test (`all-MiniLM-L6-v2`)
- ✅ MLP model (`run_mlp.py`)
- ✅ GNN model (`run_gnn.py`)
- ✅ Transformer-based model (`run_transformer.py`)
- ✅ Auto re-installation of PyG components:
  - `torch-scatter==2.0.8`
  - `torch-sparse==0.6.12`
  - `torch-cluster==1.5.9`
  - `torch-spline-conv==1.2.1`
  - `torch-geometric==2.0.4`
- ✅ UTF-8-safe requirements loader
- ✅ Full environment summary logger

---

## 📁 File structure

```
llm-gnn-slurm-template/
├── requirements.txt                # Initial pip requirements
├── requirements_utf8.txt          # Auto-converted to UTF-8
├── submit_pipeline.slurm          # ✅ Main SLURM script
├── check_llm_gpu.py               # Test GPU + SentenceTransformer
├── check_env.py                   # Print versions of major packages
├── run_mlp.py                     # Minimal MLP model test
├── run_gnn.py                     # Minimal GNN model test
├── run_transformer.py             # Minimal transformer test
├── Multitask_Deep_Learning_LLM_IMDB_method_v1_5.py  # Your actual training script
├── check_env_log.txt              # Output of check_env.py
└── README.md
```

---

## 🚀 How to Run

Submit the SLURM script with:

```bash
sbatch submit_pipeline.slurm
```

It will:

1. Load modules (`pytorch-extra-py39-cuda11.2-gcc9`, etc.)
2. Upgrade pip
3. Convert `requirements.txt` to UTF-8 if needed
4. Install `sentence-transformers` if missing
5. Remove old PyG `.so` binaries
6. Reinstall all PyG dependencies using `--prefer-binary`
7. Run: `check_llm_gpu.py`, `run_mlp.py`, `run_gnn.py`, `run_transformer.py`
8. Save environment info to `check_env_log.txt`

---

## 🧪 Tested On

- ✅ CUDA 11.2
- ✅ PyTorch 1.9.1
- ✅ Python 3.9
- ✅ NVIDIA A100 GPU

---

## 🧩 Optional Extensions

- Add job array support for hyperparameter tuning
- Integrate WandB, MLflow, or TensorBoard
- Switch to torch 2.x or CUDA 12.x when PyG builds are available

---

## 📄 License

MIT (or update with your institution's license policy)

```sh
#!/bin/bash
#SBATCH --nodelist=node004
#SBATCH --job-name=llm_gnn_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=20-00:00:00
#SBATCH --output=logs/llm_pipeline_%j.log

echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "======================"

# 🔧 Load required modules
module purge
module load pytorch-extra-py39-cuda11.2-gcc9
module load tensorflow2-py39-cuda11.2-gcc9/2.7.0
module load ml-pythondeps-py39-cuda11.2-gcc9/4.8.1

# ⬆️ Upgrade pip
pip install --upgrade --user pip

# 📦 Ensure requirements are UTF-8 encoded
REQ_FILE="requirements_utf8.txt"
if [ ! -f "$REQ_FILE" ]; then
  iconv -f ISO-8859-1 -t UTF-8 requirements.txt -o "$REQ_FILE"
fi

# 📥 Install standard Python packages if needed
if ! python3 -c "import sentence_transformers" &> /dev/null; then
  echo "🔄 Installing dependencies from $REQ_FILE..."
  pip install --user -r "$REQ_FILE"
else
  echo "✅ sentence-transformers already installed."
fi

# 🧹 Remove old PyG versions
echo "🧹 Removing old torch-geometric packages..."
rm -rf ~/.local/lib/python3.9/site-packages/torch_sparse*
rm -rf ~/.local/lib/python3.9/site-packages/torch_scatter*
rm -rf ~/.local/lib/python3.9/site-packages/torch_cluster*
rm -rf ~/.local/lib/python3.9/site-packages/torch_spline_conv*
rm -rf ~/.local/lib/python3.9/site-packages/torch_geometric*

# 📦 Install PyG with CUDA 11.2 wheels using --prefer-binary
echo "📦 Installing PyG binary wheels..."
pip install --no-cache-dir --prefer-binary --user \
  torch-scatter==2.0.8 \
  torch-sparse==0.6.12 \
  torch-cluster==1.5.9 \
  torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.9.1+cu112.html

pip install --no-cache-dir --prefer-binary --user torch-geometric==2.0.4

# 🚀 Run main pipeline scripts
echo "🚀 Running LLM + GNN pipeline..."
python check_llm_gpu.py
python run_mlp.py
python run_gnn.py
python run_transformer.py
python check_env.py > check_env_log.txt

echo "✅ Job complete."
```
