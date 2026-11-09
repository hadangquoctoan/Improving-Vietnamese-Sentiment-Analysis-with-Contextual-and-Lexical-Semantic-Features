# Vietnamese sentiment foundation

Research foundation for Vietnamese sentiment classification: PhoBERT contextual embeddings fused with an external 12-dimensional lexicon branch (polarity, VAD, eight emotions). Evaluated on UIT-VSFC (3-class), VLSP2016 (3-class) and AIVIVN 2019 (binary).

> Đây là code nghiên cứu, không phải sản phẩm production. Số liệu chỉ có giá trị với đúng dataset, split, seed và checkpoint ghi trong `artifacts/reports/`.

## Pipeline

```text
text (đã word-segment)
  ├─ PhoBERT (slow tokenizer, use_fast=False) → H [B,T,768]
  └─ LexiconStore (12D, mask theo từng chiều) → L [B,T,12] + presence
        → LexiconProjection → L̃ [B,T,768]
  g = σ(W_g [H; L̃] + b)                  # gate 768 chiều
  F = H + g ⊙ L̃                          # residual, OOV → F = H tuyệt đối
  masked mean pooling → z [B,768]
  classifier → logits [B,3] (UIT/VLSP) hoặc [B,2] (AIVIVN)
```

- Mode `fused` (mặc định khi train/infer): chạy cả hai nhánh.
- Mode `contextual`: chỉ PhoBERT, dùng cho ablation.
- Lexicon lookup miss → zero vector + zero presence; special tokens giữ nguyên `H`.

## Kết quả hiện có

| Dataset | Model | Accuracy | Macro-F1 | Report |
|---|---|---:|---:|---|
| UIT-VSFC | PhoBERT + lexicon fusion | 0,9409 | 0,8451 | `artifacts/reports/test_eval.json` |
| VLSP2016 | zero-shot từ UIT | 0,5333 | 0,5123 | `artifacts/reports/vlsp2016_test_best.json` |
| VLSP2016 | fine-tune trên VLSP train | 0,7524 | 0,7532 | `artifacts/reports/vlsp2016_test_finetuned.json` |
| AIVIVN 2019 | binary, train trên AIVIVN | 0,9148 | 0,9131 | `artifacts/reports/aivivn2019_test.json` |

**Caveat:**
- AIVIVN test là split nội bộ (toàn bộ ID `train_*`, 8 văn bản trùng train sau split), không phải public test chính thức.
- VLSP2016 test đã được dùng nhiều lần; các kết quả VLSP chỉ nên xem là exploratory, không phải holdout sạch.
- UIT-VSFC test đã dùng một lần cho pipeline hiện tại.
- Không dùng test để chọn threshold/epoch/lexicon.

## Cấu trúc repo

```text
src/vnsentiment/
├── preprocessing/      # NFC/lowercase, variant map, VnCoreNLP segmenter
├── phobert_branch/     # tokenizer, encoder, BackboneBatch (word_ids)
├── lexicon/            # LexiconStore, features/alignment, profile, CLI
├── models/             # SentimentModel
│   └── fusion/         # projection, gate, residual, pooling, classifier
├── training/           # data loaders, train step, loop, metrics, evaluate, factory
├── contracts/          # BackboneBatch, LexicalBatch, TrainBatch
├── inference.py        # SentimentPredictor
└── inspection.py       # CLI diagnostics cho preprocessing/PhoBERT

scripts/                # prepare/train/benchmark/build/figures
tests/                  # unit + integration
data/                   # raw/processed/lexicons (gitignored)
artifacts/              # checkpoints/reports/logs/cache (gitignored)
plans-and-report/       # Obsidian vault: plans, sessions, index (gitignored)
```

## Môi trường

- Python >= 3.12, Conda env `nlpdevelop`.
- `D:/anaconda/envs/nlpdevelop/python.exe` hoặc `conda activate nlpdevelop`.
- Dependencies: `torch>=2.0`, `transformers>=4.40`, `pytest>=8.0`.
- GPU hiện tại: Quadro RTX 3000 6 GB, `torch 2.14.1+cu130`.
- Không tạo `.venv` trong workspace.

```bash
D:/anaconda/envs/nlpdevelop/python.exe -m pytest tests/ -q
```

Kết quả hiện tại: **262 passed, 5 skipped, 22 subtests**.

## Lệnh chính

```bash
# Kiểm tra preprocessing + lexicon + PhoBERT hidden states
D:/anaconda/envs/nlpdevelop/python.exe main.py

# UIT-VSFC (3 lớp)
D:/anaconda/envs/nlpdevelop/python.exe scripts/train_sentiment.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/evaluate_test.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/predict_sentiment.py

# VLSP2016 (3 lớp)
D:/anaconda/envs/nlpdevelop/python.exe scripts/prepare_vlsp2016.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/train_vlsp2016.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/benchmark_vlsp2016.py

# AIVIVN 2019 (binary)
D:/anaconda/envs/nlpdevelop/python.exe scripts/prepare_aivivn2019.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/train_aivivn2019.py
D:/anaconda/envs/nlpdevelop/python.exe scripts/benchmark_aivivn2019.py

# Ảnh benchmark cho slides (PNG 200 dpi + SVG)
D:/anaconda/envs/nlpdevelop/python.exe scripts/make_benchmark_figures.py
```

## Demo (Report 4)

Giao diện web + API chạy các checkpoint đã train: nhấp đúp `demo/START_DEMO.bat`, mở `http://127.0.0.1:8000`. Hướng dẫn nhanh: [`DEMO.md`](DEMO.md). Cài đặt chi tiết, tài nguyên cần tải và hợp đồng API: [`demo/README.md`](demo/README.md).

## Lexicon

- Composite frozen: `data/lexicons/composite_12d/v0.2.0/` — 9.525 entries.
- Mở rộng offline:
  - `v0.3.2` — WordNet synonym + guard VAD: 9.791 entries; polarity train 23,70% → 27,45%.
  - `v0.3.3` — thêm fastText kNN + guard: 9.888 entries; polarity train 23,70% → 27,86%.
  - `v0.3.0`/`v0.3.1` là bản trước guard, chỉ giữ làm bằng chứng audit.
- `factory.build_sentiment_model(..., lexicon_path=...)` và các CLI train/eval/inference đã hỗ trợ `--lexicon-path`; mặc định vẫn là `v0.2.0`.
- Build lại:
  ```bash
  D:/anaconda/envs/nlpdevelop/python.exe scripts/build_expanded_lexicon.py \
    --version 0.3.2 --output-dir data/lexicons/composite_12d/v0.3.2
  ```

## Tài liệu

- Dashboard: `plans-and-report/00-INDEX.md`.
- Plans: `plans-and-report/01-PLANS/`.
- Session logs: `plans-and-report/02-FEATURE-SESSIONS/`.
- Kiến trúc hai nhánh: `docs/Part2/figures/two_branch_adaptive_fusion.png` / `.svg`.

## License dữ liệu

UIT-VSFC, VLSP2016, AIVIVN 2019, VietSentiWordNet, NRC-VAD, VnEmoLex và Vietnamese WordNet chỉ dùng cho nghiên cứu; **không tái phân phối**. `data/` và `artifacts/` không được commit.
