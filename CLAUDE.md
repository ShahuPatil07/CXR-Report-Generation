# MedReport-VLM — Project Context

## Goal
Fine-tune a vision-language model (VLM) to generate structured radiology reports
from chest X-ray images. Combines LLM fine-tuning, CV encoder alignment, and 
RLHF/DPO alignment. Built as a summer portfolio project

## Papers
- CheXagent — vision-language alignment for CXR
- BioViL-T — temporal/contextual radiology VLM
- UniRG (Microsoft, 2026) — RL-aligned report generation

## Dataset
- NLM-CXR (OpenI / Indiana University) — open access, no credentialing
- ~7k real image-report pairs
- Augmented to ~20k via:
  - Option A: report augmentation (prompt Claude/BioMistral to rephrase 
    existing reports 2-3x, keeping same findings)
  - Option B: CheXpert label → structured finding → natural prose synthesis
  - RoentGen (pretrained diffusion, inference-only) for rare-class CXR synthesis
- Test set: 100% real, unaugmented pairs only

## Architecture
- Base model: LLaVA-Med or CheXagent (HuggingFace)
- ViT encoder: FROZEN throughout training (too little data to update)
- LLM backbone: QLoRA fine-tuned (4-bit, rank 8 adapters)
- Week 3: DPO alignment using ground-truth=chosen, model-gen=rejected pairs

## Evaluation suite
- BLEU, ROUGE (proxy metrics)
- ClinicalBERT F1
- RadGraph F1 (entity + relation)
- GREEN score
- GradCAM attention rollout (visual grounding, Week 4)
- Multi-label classification head on ViT features — 14 CheXpert labels

## Weekly milestones
- Week 1: data pipeline, augmentation, base model inference, eval baseline
- Week 2: QLoRA fine-tuning, instruction tuning, cross-modal ablations
- Week 3: DPO/RLHF alignment, reward model from ClinicalBERT + RadGraph F1
- Week 4: GradCAM grounding, classification head, final eval, 4-page write-up

## Key decisions already made
- No GAN training from scratch (unstable on 7k images, too time-consuming)
- RoentGen used inference-only, not retrained
- LoRA rank 8 (not 16) due to small dataset
- DPO preferred over PPO for simplicity; PPO as stretch goal
- Experiment tracking: Weights & Biases

## Tooling
- Claude Code as pair-programmer (CLI, not API)
- HuggingFace Transformers + PEFT for LoRA
- Python, PyTorch
- Compute: single A100 or L4 (Colab Pro+ or equivalent)