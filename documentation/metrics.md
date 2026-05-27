# `src/metrics.py` — Evaluation Metrics

## Overview

```python
from src.metrics import evaluate_all, print_metrics

metrics = evaluate_all(hypotheses, references)
# Returns: {bleu1, bleu4, rouge1, rouge2, rougeL, bertscore_p, bertscore_r, bertscore_f1}

print_metrics(metrics)
```

`hypotheses` = list of generated reports (strings)  
`references` = list of ground-truth reports (strings), same order

---

## BLEU — `compute_bleu`

**What it measures:** n-gram overlap between hypothesis and reference.

BLEU-1 counts how many individual words match. BLEU-4 requires 4-word phrases to match exactly. BLEU-4 is the standard in NLP papers because it penalizes fluent-sounding but wrong text.

**How it's computed:**
```python
corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
```
Corpus-level BLEU is computed across all samples at once (not averaged per-sample). This is standard and avoids inflating the score on short outputs.

**What numbers to expect:**

| Scenario | BLEU-4 |
|---|---|
| LLaVA-1.5 zero-shot | 0.01 – 0.05 |
| After QLoRA fine-tuning | 0.08 – 0.18 |
| After DPO alignment | 0.12 – 0.22 |

**Limitation for medical text:** BLEU penalizes synonyms harshly. "Heart is enlarged" and "Cardiomegaly present" mean the same thing clinically but score near zero overlap. This is why BLEU alone is not enough — you also need BERTScore.

---

## ROUGE — `compute_rouge`

**What it measures:** recall-oriented overlap. ROUGE-L measures the longest common subsequence.

The function returns mean F1 across all samples for `rouge1`, `rouge2`, and `rougeL`. ROUGE-L is the primary number to watch — it captures fluency better than ROUGE-1.

**How it's computed:**
```python
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
```
`use_stemmer=True` means "lungs" and "lung" are treated as the same word. This is the right setting for medical text.

**What numbers to expect:**

| Scenario | ROUGE-L |
|---|---|
| LLaVA-1.5 zero-shot | 0.08 – 0.15 |
| After QLoRA fine-tuning | 0.18 – 0.30 |
| After DPO alignment | 0.22 – 0.35 |

**In `finetune.py`**, ROUGE-L is used as the checkpoint selection criterion (`best_rougeL`). This is a reasonable choice — it correlates better with clinical quality than BLEU.

---

## BERTScore — `compute_bertscore`

**What it measures:** semantic similarity using contextual embeddings. Two reports can have different words but similar meanings, and BERTScore will give them a high score.

**Why SciBERT?** Standard BERTScore uses `bert-base-uncased`. SciBERT (`allenai/scibert_scivocab_uncased`) was trained on 1.14M scientific papers and has a medical vocabulary. It understands that "pleural effusion" and "fluid in the pleural space" are semantically close.

```python
_CLINICAL_MODEL = "allenai/scibert_scivocab_uncased"
P, R, F1 = bert_score.score(hypotheses, references, model_type=_CLINICAL_MODEL)
```

Returns precision, recall, and F1. Report F1 (`bertscore_f1`) as your headline metric for clinical relevance.

**What numbers to expect:**

| Scenario | BERTScore F1 |
|---|---|
| LLaVA-1.5 zero-shot | 0.80 – 0.84 |
| After QLoRA fine-tuning | 0.84 – 0.88 |
| After DPO alignment | 0.86 – 0.90 |

BERTScore starts relatively high even for bad models because all medical text shares vocabulary. The delta matters more than the absolute number.

**Note:** BERTScore is slow (~30 seconds for 200 samples on GPU). This is why `evaluate_all` has a `skip_bertscore=True` flag used during training validation. Only compute it at the end.

---

## For Week 4: adding ClinicalBERT F1 and RadGraph F1

The CLAUDE.md mentions two additional metrics that are more clinically meaningful:

### ClinicalBERT F1
Replace `allenai/scibert_scivocab_uncased` with `emilyalsentzer/Bio_ClinicalBERT`. This model was trained on clinical notes (MIMIC-III) rather than scientific papers — closer to radiology language.

```python
metrics.update(compute_bertscore(hyps, refs, model_type="emilyalsentzer/Bio_ClinicalBERT"))
```

### RadGraph F1
RadGraph extracts clinical entities (findings, anatomy) and relations from radiology reports and compares them. It's the gold standard metric for this task.

Install: `pip install radgraph`  
Usage: see https://github.com/StanfordAIMI/radgraph

RadGraph F1 will be lower than BERTScore F1 but is more meaningful — it penalizes hallucinated findings that sound plausible but aren't in the reference.

### GREEN score
GREEN (Generative Radiology Evaluation and Error Notation) uses an LLM to score reports by counting clinically significant errors. Paper: https://arxiv.org/abs/2405.03595

This is the most informative metric for the final write-up, but it requires API calls and is expensive to run on large sets. Run it on 50–100 samples for the final evaluation.

---

## Using metrics to make decisions

| Situation | Which metric to trust |
|---|---|
| Picking the best checkpoint during training | ROUGE-L (fast, no GPU needed) |
| Comparing zero-shot vs fine-tuned | BLEU-4 + ROUGE-L + BERTScore F1 |
| Comparing fine-tuned vs DPO-aligned | BERTScore F1 + RadGraph F1 |
| Final write-up headline number | GREEN + RadGraph F1 |
