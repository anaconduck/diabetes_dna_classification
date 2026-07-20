# Adaptively Weighted Multi-scale k-mer Representation with Scale Attention for Type 2 Diabetes Classification

This repository contains the official implementation of the deep learning architecture for identifying Type 2 Diabetes (T2D) signatures directly from raw, unaligned DNA sequences. 

Unlike traditional methods that rely on static Bag-of-K-mers (which destroy spatial sequence topology), this pipeline introduces a **Token-level Scale Attention** mechanism. It dynamically learns to weight and fuse multi-scale Word2Vec representations ($k=3, 4, 5$) at every nucleotide position, processed sequentially through a temporally-masked LSTM network.

## 🚀 Key Novelties
1. **Multi-Scale k-mer Extraction:** Extracts overlapping $k=3, 4, 5$ motifs to capture both local binding affinities and broad regulatory contexts.
2. **Dense Word2Vec Representation:** Learns semantic DNA representations using Continuous Bag-of-Words (CBOW) rather than sparse one-hot encoding.
3. **Token-Level Scale Attention:** A dynamic neural lens that calculates softmax attention weights across scales per-nucleotide, mirroring biological transcription factor scanning.
4. **Sequence-Preserving LSTM:** Utilizes recurrent masking to handle variable DNA lengths, capturing long-range genomic dependencies without gradient degradation.

## 📂 Project Structure

- `configs/`: Contains `config.yaml` to orchestrate all hyperparameters (k-mer sizes, LSTM units, Word2Vec dimensions, etc.) without hardcoding.
- `data/`: Directory for placing the `dataset.csv` (raw DNA sequences).
- `outputs/`: Automatically stores classification metrics, ROC-AUC reports, and attention weight logs.
- `src/`: 
  - `adaptive_kmer/`: Core logic for multi-scale representation and token-level Scale Attention.
  - `datasets/`: Stratified loading and Random Undersampling for class balance.
  - `encoding/`: K-mer tokenization and Word2Vec CBOW training.
  - `models/`: TensorFlow/Keras definitions for the Masked LSTM and Attention architectures.
  - `evaluation/`: 5-Fold Stratified Cross-Validation and metric calculation.
- `weights/`: Stores the trained LSTM `.h5` model and `.model` Word2Vec binaries.

## ⚙️ Setup & Installation

1. Clone the repository and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your raw genomic data (`dataset.csv`) inside the `data/` directory.


## 🔬 Reproducing the Journal Experiments
To strictly reproduce the analytical tables presented in the published manuscript, you can run the individual stage scripts in the following order:

**Stage 1: Impact of Imbalance Strategies**
```bash
python imbalance_study.py
```
*(Note: After determining the optimal balancing strategy from this stage, ensure you update the `imbalance_strategy` field in `configs/config.yaml` before running subsequent stages).*

**Stage 2: Baseline Model Comparisons**
```bash
python compare_models.py
```

**Stage 3: Architectural Ablation Study**
```bash
python ablation_study.py
```

**Stage 4: Token-Level Attention Analysis**
```bash
python attention_analysis.py
```

**Stage 5: Robustness via 5-Fold Cross Validation**
```bash
python robustness_cv.py
```

All generated metrics, confusion matrices, and attention distribution logs will automatically be saved and formatted in the `outputs/` directory.

## 🔒 Data Availability
Please note that the raw genomic dataset (`dataset.csv`) is **strictly confidential** and has been explicitly excluded from this repository via `.gitignore` to comply with medical data privacy standards. 

If you require access to the dataset for verification or academic collaboration purposes, please send a formal request via email to: **wahyukusumaw29@gmail.com**
