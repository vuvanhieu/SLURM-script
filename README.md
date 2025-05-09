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

# 🚀 LLM + GNN Multimodal Pipeline (HPC SLURM-Compatible)

This repository provides a ready-to-run, SLURM-compatible configuration for executing a multimodal deep learning pipeline integrating:

- ✅ PyTorch (CUDA 11.2)
- ✅ SentenceTransformers
- ✅ Torch Geometric (GNN)
- ✅ TensorFlow (GPU check)
- ✅ HuggingFace Hub
- ✅ MLP, GNN, and Transformer-based model runs

---

## 📂 Repository Structure

```
.
├── requirements.txt              # Core Python dependencies
├── requirements_utf8.txt         # Auto-generated UTF-8 copy
├── check_llm_gpu.py              # Checks GPU and LLM functionality
├── run_mlp.py                    # Simple MLP model demo
├── run_gnn.py                    # GNN model using torch_geometric
├── run_transformer.py           # Transformer inference with SentenceTransformer
├── check_env.py                 # Debug versions of major packages
├── setup_once.slurm             # First-time SLURM setup and run script
├── run_only.slurm               # Lightweight SLURM script for re-runs
└── logs/
    └── llm_pipeline_<job_id>.log
```

---

## ⚙️ SLURM Script Usage Guide

### `setup_once.slurm`
> **Use this script only for the first run** (or after environment reset)

- Converts `requirements.txt` to UTF-8 (if needed).
- Installs all dependencies via pip.
- Removes broken or incompatible PyG packages.
- Installs `torch-scatter`, `torch-sparse`, `torch-cluster`, and `torch-spline-conv` from PyG binary wheels.
- Ensures compatibility with `torch==1.9.1 + cu112`.
- Runs the full pipeline and records logs.

### `run_only.slurm`
> **Use this script for all subsequent runs**

- Skips all installation and cleanup.
- Directly runs model scripts and logs results.
- Faster and suitable for multiple experiments.

---

## 📋 Output Logs

All SLURM outputs are written to:

```
logs/llm_pipeline_<job_id>.log
```

Environment check outputs are stored in:

```
check_env_log.txt
```

---

## 💡 Example SLURM Submission

```bash
sbatch setup_once.slurm     # For first-time setup
sbatch run_only.slurm       # For subsequent clean runs
```

---

## 🔧 Notes

- Ensure you are running on a compatible HPC node (CUDA 11.2, PyTorch 1.9.1).
- If you upgrade or change `torch`, re-run `setup_once.slurm` to refresh PyG.
- The pipeline is modular: you can plug in your own `run_*.py` models.

---

## 🧪 Verified Compatibility

- NVIDIA A100-SXM4 (CUDA 12.6 runtime, PyTorch 1.9.1 + cu112 compiled)
- PyG 2.0.4 with binary wheels (avoids source compilation errors)
- TensorFlow 2.7.0 GPU detection (optional)

---

## 🧠 Authors

This template was created for reproducible HPC workflows using LLMs + GNNs with efficient deployment and debugging support.

Feel free to customize it for your experiments.

### sbatch run_only.slurm       # For subsequent clean runs
```sh
#!/bin/bash
#SBATCH --nodelist=node004
#SBATCH --job-name=llm_gnn_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=20-00:00:00
#SBATCH --output=logs/llm_pipeline_run_%j.log

echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "======================"

# ✅ Load environment (no reinstall)
module purge
module load pytorch-extra-py39-cuda11.2-gcc9
module load tensorflow2-py39-cuda11.2-gcc9/2.7.0
module load ml-pythondeps-py39-cuda11.2-gcc9/4.8.1

# 🏃‍♂️ Run only the core pipeline
echo "🚀 Running LLM + GNN pipeline..."
python check_llm_gpu.py
python run_mlp.py
python run_gnn.py
python run_transformer.py
python check_env.py > check_env_log.txt

echo "✅ Job complete."
```
