# Week 3 and Week 4 — Build Guide

This guide explains the concepts you need to understand and the structure of what to build. It does not give you the code — you write that yourself. Each section ends with a self-check question. If you can answer it, you're ready to code.

---

# Week 3 — DPO Alignment

## What problem DPO solves

After QLoRA fine-tuning, your model generates grammatical, domain-relevant reports. But it can still:
- Hallucinate findings ("left pleural effusion" when the lungs are clear)
- Be vague where it should be specific ("some abnormality noted" instead of naming it)
- Produce a fluent-sounding report that misses the key finding entirely

ROUGE and BERTScore don't catch these failures because they measure word overlap, not clinical correctness. The model needs feedback on *quality*, not just *fluency*.

## The DPO idea

DPO is not reinforcement learning in the traditional sense — there is no reward model, no policy rollout, no value function. Instead, it directly optimizes a preference: given two outputs for the same input, make the model more likely to produce the "chosen" one and less likely to produce the "rejected" one.

The loss function looks like this conceptually:

```
L = -log(sigmoid( beta * (log P(chosen|input) - log P(rejected|input)) ))
               ─────────────────────────────────────────────────────────
               this term is large when model prefers chosen over rejected
```

`beta` controls how strongly the model is penalized for drifting away from the SFT checkpoint. Low beta = more aggressive alignment. High beta = conservative, stays close to the SFT model.

**Why not PPO?** PPO requires a separate reward model trained to score outputs. That reward model needs its own training data, is expensive to train, and can be gamed by the policy (reward hacking). DPO skips all of this — the "reward" is implicit in the (chosen, rejected) pairs themselves. For a 4-week project, DPO is the right call.

**The key insight for this project:** you already have a natural source of (chosen, rejected) pairs.
- **chosen** = the real ground-truth report from NLM-CXR (always better than a model output)
- **rejected** = what your fine-tuned model generates for the same image (plausible but imperfect)

You don't need human annotators.

## Step 1: Generate the rejected responses — `generate_dpo_data.py`

**Concept:** Run your fine-tuned model (from Week 2's `best_checkpoint`) on the *training set* with `do_sample=True` and a temperature around 0.8. This gives diverse, slightly noisy outputs — not the greedy best guess. You want the rejected samples to be plausible-but-wrong, not obviously terrible.

**What this file should do:**
1. Load the Week 2 best checkpoint (base model + PEFT adapters — look at `finetune.md` for how to load a checkpoint)
2. Load only the training split (never generate rejected samples from the val or test set)
3. For each training sample, generate one report with sampling
4. Filter out any generated report that happens to be identical to the ground truth
5. Save the result as a JSON list where each entry has four fields: `image`, `prompt`, `chosen`, `rejected`

**The prompt field** should be the same template you used in inference — the part *before* "ASSISTANT:", i.e., `EVAL_TEMPLATE` from `src/dataset.py`.

**Important:** generation will take a long time (roughly 1–2 seconds per image × 2,600 images = ~1.5 hours on an A100). Run it overnight or generate a subset (1,000 pairs is enough for DPO).

**Self-check:** Why do you use `do_sample=True` for generating rejected samples instead of greedy decoding (`do_sample=False`)? What would happen to your DPO training if all rejected samples were generated greedily?

---

## Step 2: Write `dpo_align.py`

**Library to use:** TRL (`pip install trl`). The class is `DPOTrainer` and its config is `DPOConfig`. Read the TRL docs: https://huggingface.co/docs/trl/dpo_trainer

**What this file should do:**
1. Load the `dpo_pairs.json` file from Step 1
2. Convert it to a HuggingFace `Dataset` object (from the `datasets` library)
3. Load the Week 2 best checkpoint as the starting model (same as `generate_dpo_data.py`)
4. Configure `DPOConfig` with the right hyperparameters (see below)
5. Instantiate `DPOTrainer` and call `.train()`
6. Save the final checkpoint to `results/dpo/`

**Key hyperparameters — understand these before setting them:**

| Parameter | What it controls | Good starting value |
|---|---|---|
| `beta` | KL penalty — how far DPO can move from SFT model | 0.1 (loosen to 0.05 if improvement is small, tighten to 0.3 if outputs degrade) |
| `learning_rate` | Should be lower than SFT LR — you're making fine adjustments | 5e-5 (one-quarter of your SFT LR) |
| `num_train_epochs` | DPO converges fast — 1 epoch is usually enough | 1 |
| `ref_model` | The reference model DPO compares against | Set to `None` — TRL handles this with PEFT automatically |

**Why `ref_model=None` works:** DPO needs to compare the current model against a frozen reference (the SFT model). Normally you'd need two model copies (14 GB). When `ref_model=None` and your model is a PEFT adapter, TRL is smart enough to compute the reference logits by running the *base* model (without adapters) — which it already has in memory. This halves your VRAM requirement.

**The multimodal gotcha:** `DPOTrainer` was originally designed for text-only models. For LLaVA (image + text), you need to make sure the `pixel_values` from the image are properly included in the batch. Check whether TRL's latest version supports multimodal DPO natively (it started being added in TRL ~0.9+). If it doesn't, you'll need to write a custom data collator that adds `pixel_values` to each batch. This is the hardest part of Week 3 — don't skip reading the TRL changelog.

**Self-check:** What is the reference model in DPO and why does it exist? What would happen if you ran DPO without a KL penalty (beta=0)?

---

## Step 3: Evaluate DPO improvement

After training, run `baseline.ipynb` (or a copy of it) three times, each time pointing `MODEL_ID` at a different checkpoint:
1. `llava-hf/llava-1.5-7b-hf` (zero-shot base)
2. `results/finetune/<run>/best_checkpoint` (after SFT)
3. `results/dpo/` (after DPO)

Compute BLEU-4, ROUGE-L, BERTScore F1 for all three. That three-row table is the core result of this project.

Expected pattern: SFT should jump significantly over base. DPO should add a smaller but meaningful improvement, particularly on BERTScore F1 (since DPO targets quality, not just fluency).

---

---

# Week 4 — GradCAM, Classifier Head, Final Eval

## Part A: Attention Rollout — `gradcam.ipynb`

### Concept

CLIP ViT-L processes an image by dividing it into a grid of 16×16 pixel patches (the input image is 336×336, so you get a 24×24 = 576 patch grid). Each patch becomes one token. The ViT runs 24 transformer layers, and each layer has multi-head self-attention — every patch attends to every other patch.

**Attention rollout** is a technique to aggregate attention across all 24 layers into a single heatmap showing which patches were most influential for the final representation. The idea: if layer 1 attends from patch A → B, and layer 2 attends from patch B → C, then patch A transitively influenced C. Rollout multiplies attention matrices across layers to capture these transitive paths.

The formula is:
```
Rollout = A_1 · A_2 · ... · A_24
```
where each `A_i` is the mean attention matrix (averaged over heads) for layer `i`, with identity added: `A_i = 0.5 * A_i + 0.5 * I` (the identity term represents residual connections).

### What to build

Create `gradcam.ipynb` with these steps:

**Step 1 — Hook into the ViT attention layers**

PyTorch `register_forward_hook` lets you intercept the output of any module during a forward pass. You need to attach hooks to every self-attention layer in the ViT and collect the raw attention weight matrices.

To find the right module path, run this after loading your model:
```python
for name, module in model.named_modules():
    print(name)
```
Look for the repeated pattern of attention layers inside `vision_tower`. The attention weights are usually in the second element of the attention output tuple.

**Step 2 — Collect and compute rollout**

After running a forward pass with the hooks active, you'll have a list of 24 attention matrices, each of shape `(batch, num_heads, seq_len, seq_len)`. Average over heads first, then apply the rollout formula above. The result is a `(seq_len, seq_len)` matrix. Take the first row (index 0 = [CLS] token) and drop the first element — the remaining 576 values are the importance of each patch.

**Step 3 — Reshape and overlay**

Reshape the 576-element vector into a `(24, 24)` grid. Resize it to match the original image size using bilinear interpolation. Overlay on the CXR image with a colormap (e.g., `"hot"` or `"jet"`) at ~50% transparency.

**Step 4 — Pick good examples**

Don't show random images. Go back to your predictions CSV from `results/baseline/predictions.csv` and find:
- A case where the model correctly identified "pleural effusion" → does the heatmap highlight the costophrenic angles?
- A case with "cardiomegaly" → does it highlight the cardiac silhouette?
- A case where the model was wrong → where was it looking?

These three cases tell a story.

**Key libraries:** `torch.nn.Module.register_forward_hook`, `numpy`, `PIL.Image.resize(mode=Image.BILINEAR)`, `matplotlib`

**Self-check:** The ViT's sequence includes a [CLS] token plus patch tokens. Why do you take the [CLS] row of the rollout matrix rather than averaging all rows? What does the [CLS] token represent in ViT?

---

## Part B: 14-Label CheXpert Classifier Head — `classifier.py`

### Concept

You already have a frozen ViT encoder that produces rich visual features for each CXR. Instead of only using these features for report generation (through the projection MLP into the LLM), you can attach a lightweight MLP classifier directly to the ViT output to predict which of the 14 CheXpert pathologies are present.

This is **multi-label classification** — a single image can have multiple labels simultaneously (e.g., both pleural effusion AND cardiomegaly). The output is not a softmax over 14 classes — it's 14 independent sigmoid outputs, each predicting "present/absent" for one pathology.

The loss function is **binary cross-entropy** applied to all 14 outputs independently.

### Step 1: Get the labels

NLM-CXR doesn't have CheXpert labels, so you need to auto-generate them from the report text.

Install: `pip install chexpert-labeler`

Run the labeler on your `report_text` column. It parses each report and outputs a 14-element vector per report where values are: 1 (positive), 0 (negative), -1 (uncertain), NaN (not mentioned). You'll need to decide how to handle uncertain labels — common approaches are to treat uncertain as positive (pessimistic) or drop them (conservative). Start with dropping them.

Add the labels as new columns to your DataFrame and save a new `cxr_dataset_labeled.csv`.

### Step 2: Build the classifier

Create a `CheXpertClassifier` `nn.Module` class that takes ViT image features as input and outputs 14 logits. Architecture:
- Input: pooled ViT features (mean over patch tokens, not [CLS]). The CLIP ViT-L hidden size is 1024.
- Hidden layer: 512 units, ReLU, Dropout(0.2)
- Output: 14 units, no activation (apply sigmoid separately during evaluation)

The ViT is frozen — you do not update it. Only the classifier MLP is trained.

### Step 3: Extract ViT features

During training you don't want to run the full LLaVA model for every batch. Instead, pre-extract ViT features once and save them:
- Load the model with frozen ViT
- For each image, run `model.vision_tower(pixel_values)` and save the output
- Store as a `.pt` tensor file per split: `train_features.pt`, `val_features.pt`, `test_features.pt`

Then build a simple Dataset that loads `(features, labels)` pairs from these cached files — this makes classifier training fast (no GPU needed for the ViT forward pass during each epoch).

### Step 4: Training loop

Write a standard PyTorch training loop:
- Loss: `nn.BCEWithLogitsLoss()` (numerically stable, applies sigmoid internally)
- Optimizer: Adam, LR 1e-3 (much higher than LLM fine-tuning — the MLP is small and randomly initialized)
- Epochs: 20–30 (fast since you're only training the MLP)
- Metric: AUC-ROC per label + macro average (use `sklearn.metrics.roc_auc_score`)

### Step 5: Evaluate

Report per-label AUC-ROC and macro AUC. Good performance is >0.80 AUC on most labels. Compare against the literature baseline for this dataset (search for "NLM-CXR classification" or "OpenI classification" on Papers with Code).

**Self-check:** Why do you use BCEWithLogitsLoss instead of CrossEntropyLoss for this problem? What's the difference, and what assumption about the output does each one make?

---

## Part C: Final Evaluation Notebook — `final_eval.ipynb`

This is a clean notebook that runs everything and produces publication-ready figures and tables.

**Structure to build:**

**Section 1 — Quantitative results table**

Run inference on the full test set (555 samples) for all three models: base, SFT, DPO. Compute BLEU-4, ROUGE-L, BERTScore F1, and AUC-ROC (from classifier). Format as a table.

**Section 2 — Qualitative comparison**

Pick 5 test images that have specific, named findings. For each image, show:
- The CXR image
- The ground-truth report
- Base model output
- SFT model output
- DPO model output

Side-by-side like this:
```
[image] | GT: "Bilateral pleural effusion..."
        | Base: "The chest X-ray is normal..."
        | SFT: "There is fluid in the pleural space..."
        | DPO: "FINDINGS: Bilateral pleural effusion with blunting of costophrenic angles..."
```

**Section 3 — Attention rollout gallery**

Call your `gradcam.ipynb` functions here. Show 3 rows × 3 columns = 9 cases. Each cell = image with heatmap overlay + the key finding word.

**Section 4 — Error analysis**

Pick 3 cases where the model is confidently wrong. What did it say? What should it have said? What is the ViT looking at? These failures are the most informative part of any ML paper.

---

## Part D: 4-Page Write-Up

**Format:** NeurIPS short paper style (2-column, 11pt). Use the Overleaf NeurIPS template.

**Section structure:**

**Abstract** (~150 words): State the task, your approach in one sentence, and your headline numbers. Example: "We fine-tune LLaVA-1.5-7B on NLM-CXR using QLoRA and DPO alignment, improving ROUGE-L from X to Y and BERTScore F1 from X to Y over the zero-shot baseline."

**1. Introduction** (~0.5 page): Why automated radiology reporting matters (radiologist shortage, consistency). What makes it hard (hallucination, domain gap between general VLMs and medical images). What you did and what you found.

**2. Related Work** (~0.25 page): CheXagent (vision-language alignment for CXR), UniRG (RL-based alignment for report generation), BioViL-T (contextual VLMs for radiology). One sentence each.

**3. Dataset** (~0.25 page): NLM-CXR description. Split strategy and why you split at report level. Note that test set is 100% real reports.

**4. Method** (~1 page):
- 4.1 Base model: LLaVA-1.5 architecture, why ViT is frozen, LoRA math (the A×B decomposition), why rank 8
- 4.2 SFT: instruction format, label masking, training details
- 4.3 DPO: how (chosen, rejected) pairs were constructed, beta value, why DPO over PPO

**5. Experiments** (~1 page):
- Table 1: three-model comparison (BLEU-4, ROUGE-L, BERTScore F1)
- Figure 1: GradCAM rollout on 3 cases (correct attention, incorrect attention, ambiguous)
- 5.1: Quantitative analysis — which metric improved most from SFT→DPO and why
- 5.2: Qualitative analysis — what kinds of errors remain

**6. Conclusion** (~0.25 page): What worked, what didn't, future directions (temporal context per BioViL-T, synthetic augmentation with RoentGen).

**References:** NLM-CXR paper, LLaVA-1.5 paper, QLoRA paper, DPO paper, CheXagent, BioViL-T, UniRG, and the metric papers (BLEU, ROUGE, BERTScore, RadGraph).

---

## Timeline

| Days | Task | Output file |
|---|---|---|
| 1–2 | Run `baseline.ipynb`, debug Colab setup | `results/baseline/metrics.json` |
| 3–6 | Run `finetune.py`, watch W&B curves, tune LR | `results/finetune/*/best_checkpoint` |
| 7 | Write + run `generate_dpo_data.py` | `dpo_pairs.json` |
| 8–9 | Write + run `dpo_align.py` | `results/dpo/` |
| 10 | Generate CheXpert labels, write + train `classifier.py` | `cxr_dataset_labeled.csv`, classifier weights |
| 11 | Write `gradcam.ipynb`, pick good examples | Figures |
| 12 | Write `final_eval.ipynb`, compile results table | `results/final/` |
| 13–14 | Write 4-page report in Overleaf | PDF |

---

## Before you start coding each component, ask yourself

**For DPO:**
- Where exactly do the gradients come from in the DPO loss? Which parameters get updated?
- What is the role of the reference model, and why does setting `ref_model=None` still work with PEFT?
- Why generate rejected samples from the *training set* and not the test set?

**For GradCAM:**
- What does each layer of the ViT compute, and why do you need to aggregate across all layers?
- What is the [CLS] token's role, and why is it a natural choice for the heatmap source row?
- What would you expect the heatmap to look like for a normal chest X-ray (no findings)?

**For the classifier:**
- Why is multi-label classification different from multi-class classification?
- Why do you pre-extract and cache ViT features instead of running the ViT on every training batch?
- What does an AUC-ROC of 0.5 mean? What does 1.0 mean?
