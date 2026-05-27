# MedReport-VLM — Documentation Index

This folder documents every script in the project so you can build Week 3 and Week 4 independently.

## Files in this folder

| Document | What it covers |
|---|---|
| [dataset.md](dataset.md) | `src/dataset.py` — data loading, splitting, PyTorch Dataset, label masking |
| [metrics.md](metrics.md) | `src/metrics.py` — BLEU, ROUGE, BERTScore and what they actually mean |
| [baseline.md](baseline.md) | `baseline.ipynb` — zero-shot inference, interpreting results |
| [finetune.md](finetune.md) | `finetune.py` — QLoRA end-to-end: quantization, LoRA, training loop |
| [data_augmentation.md](data_augmentation.md) | How to augment the dataset (text + image) and re-fine-tune |
| [week3_week4_guide.md](week3_week4_guide.md) | What to build for DPO, GradCAM, classifier head, and write-up |
| [agent.md](agent.md) | Agentic report generation — Claude orchestrator with classify/attend/draft tools |

## Project structure recap

```
CXR-Report-Generation/
├── src/
│   ├── dataset.py          data loading + PyTorch Dataset
│   └── metrics.py          BLEU / ROUGE / BERTScore
├── baseline.ipynb          Week 1 — zero-shot eval
├── finetune.py             Week 2 — QLoRA fine-tuning
├── cxr_dataset.csv         3,955 reports, 7,470 images
├── requirements.txt
└── documentation/          ← you are here
```

## Recommended reading order

If you are new to this codebase: `dataset.md` → `finetune.md` → `metrics.md` → `baseline.md`.  
For building Week 3–4 next: go straight to `week3_week4_guide.md`.  
For the agentic component: read `week3_week4_guide.md` first (you need the classifier and attention rollout), then `agent.md`.  
For augmentation: `data_augmentation.md` is self-contained.
