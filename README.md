# LLM-Transform

Code for **LLM-Transform: Transforming LLM Representations into Item Embeddings for Session-based Recommendation**.

**LLM-Transform** encodes each item card with a frozen Qwen3-Embedding-4B to obtain its language representation. A two-layer MLP, the transformation module, transforms the language representation into the behavior space to give the transformed item embedding, which is the only item embedding used for session encoding and next-item scoring. The last-item shortcut self-attention session encoder reads the transformed item embeddings of a session and adds the transformed item embedding of the last item to the self-attention representation, and the model scores the full catalog for next-item prediction.

## Included datasets

The three Amazon Reviews 2023 categories and MovieLens-20M used in the paper are included under `data/processed/`:

| Directory | Items | Train sessions | Train prefixes | Valid sessions | Test sessions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Cell_Phones_and_Accessories` | 10,630 | 50,069 | 88,418 | 4,831 | 2,676 |
| `Grocery_and_Gourmet_Food` | 17,579 | 67,156 | 149,733 | 7,874 | 7,579 |
| `Movies_and_TV` | 26,414 | 84,246 | 251,033 | 8,847 | 7,895 |
| `MovieLens20M` | 14,028 | 527,044 | 5,156,806 | 53,790 | 46,773 |

Amazon sessions use each user's first active calendar day. MovieLens sessions group ratings by user and split at gaps longer than one hour. Sessions are capped at 20 items and split chronologically into training, validation, and test, with validation/test items restricted to the training catalog. Training uses prefix augmentation. Validation and test use one last-click example per session.

## Setup

Python 3.11 is required. Install a PyTorch build for your device. The example below uses CUDA 12.8:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Run

Set `category` to any directory name in the dataset table. To run MovieLens, use `category=MovieLens20M`:

```bash
category=Cell_Phones_and_Accessories
data_dir="data/processed/$category"
language_dir="outputs/$category/language_representations"
run_dir="outputs/$category/llm_transform"
mkdir -p outputs

# Encode item cards once and reuse the frozen language representations.
python -m llmtransform encode \
  --data-dir "$data_dir" --output "$language_dir" \
  --device cuda --batch-size 16 > "outputs/encode_$category.log" 2>&1

# Train and select the checkpoint by validation HR@20.
python -m llmtransform train \
  --config configs/llm_transform.yaml \
  --data-dir "$data_dir" --language-dir "$language_dir" \
  --output "$run_dir" --device cuda > "outputs/train_$category.log" 2>&1

# Evaluate the selected checkpoint on the last-click test split.
python -m llmtransform evaluate \
  --checkpoint "$run_dir/model.pt" \
  --data-dir "$data_dir" --language-dir "$language_dir" \
  --split test --device cuda --output "$run_dir/test_metrics.json" \
  > "outputs/evaluate_$category.log" 2>&1
```

Evaluation reports **HR@20** and **MRR@20**. Hyperparameters are in `configs/llm_transform.yaml`: transformation-module MLP widths 2560 → 1280 → 256, two self-attention blocks with four heads, dropout 0.2, and initial last-item shortcut weight γ = 0.05. All four datasets use the same configuration.
The LLM encoder weights download on first use.
