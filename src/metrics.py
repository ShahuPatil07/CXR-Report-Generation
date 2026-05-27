"""
Evaluation metrics for CXR report generation.

Metrics returned by evaluate_all():
  bleu1, bleu4            – corpus-level BLEU (nltk)
  rouge1, rouge2, rougeL  – mean F1 across samples (rouge-score)
  bertscore_p/r/f1        – mean BERTScore using allenai/scibert_scivocab_uncased
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #

def compute_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> dict:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    smoother = SmoothingFunction().method1
    hyps = [nltk.word_tokenize(h.lower()) for h in hypotheses]
    refs = [[nltk.word_tokenize(r.lower())] for r in references]

    return {
        "bleu1": corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoother),
        "bleu4": corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother),
    }


# --------------------------------------------------------------------------- #
# ROUGE
# --------------------------------------------------------------------------- #

def compute_rouge(hypotheses: Sequence[str], references: Sequence[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}

    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for key in agg:
            agg[key].append(scores[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in agg.items()}


# --------------------------------------------------------------------------- #
# BERTScore
# --------------------------------------------------------------------------- #

_CLINICAL_MODEL = "allenai/scibert_scivocab_uncased"


def compute_bertscore(
    hypotheses: Sequence[str],
    references: Sequence[str],
    model_type: str = _CLINICAL_MODEL,
) -> dict:
    import bert_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P, R, F1 = bert_score.score(
            list(hypotheses),
            list(references),
            model_type=model_type,
            verbose=False,
        )

    return {
        "bertscore_p":  float(P.mean()),
        "bertscore_r":  float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #

def evaluate_all(
    hypotheses: Sequence[str],
    references: Sequence[str],
    skip_bertscore: bool = False,
) -> dict:
    """
    Compute all metrics and return a single flat dict.
    Set skip_bertscore=True for a quick sanity check (BERTScore is slow).
    """
    metrics = {}
    metrics.update(compute_bleu(hypotheses, references))
    metrics.update(compute_rouge(hypotheses, references))
    if not skip_bertscore:
        metrics.update(compute_bertscore(hypotheses, references))
    return metrics


def print_metrics(metrics: dict) -> None:
    width = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k:<{width}} {v:.4f}")
