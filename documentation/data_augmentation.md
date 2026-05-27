# Data Augmentation for CXR Report Generation

## Why augment?

You have ~2,600 training reports. That's small for fine-tuning a 7B model. The model will likely start overfitting around epoch 2. More data = longer training before overfitting = better generalization.

There are three fundamentally different augmentation strategies. Each has different tradeoffs.

---

## Strategy 1: Image-level augmentation (easiest, always use this)

Apply random transforms to existing images during training. No new data — just different views of the same image on each epoch.

**Where to add it:** in `CXRDataset.__getitem__`, before passing the image to the processor:

```python
from torchvision import transforms

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

# In __getitem__, train mode only:
if self.mode == "train":
    image = TRAIN_TRANSFORMS(image)
```

**What's safe for CXR:**
- `RandomHorizontalFlip` — safe, a mirrored chest X-ray is still clinically valid
- `RandomRotation(±5°)` — safe, real X-rays are sometimes slightly rotated
- `ColorJitter(brightness, contrast)` — safe, represents different exposure settings
- `RandomAffine(translate=0.05)` — safe, small shifts are normal

**What's NOT safe:**
- `RandomVerticalFlip` — an upside-down chest X-ray is meaningless
- `RandomCrop` — can remove clinically important structures (lung bases, cardiac silhouette)
- `Strong rotation (>10°)` — changes apparent position of findings

**Effect:** ~1.5–2x improvement in effective data diversity per epoch. Zero extra storage. Free performance gain.

---

## Strategy 2: Report text augmentation (medium effort, high reward)

Use an LLM to rephrase existing reports 2–3 times, keeping the same clinical findings but varying the writing style. This expands your 2,600 training reports to ~7,800 without touching images.

**Why this works:** the NLM-CXR reports all come from one institution and have very uniform phrasing. The model can overfit to specific sentence patterns ("The lungs are clear and expanded") rather than learning to identify findings. Rephrasing teaches the model that the same finding can be expressed many ways.

### Implementation

Create `augment_reports.py`:

```python
import anthropic
import pandas as pd
from pathlib import Path

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

REPHRASE_PROMPT = """You are a radiologist. Rephrase the following radiology report section 
in a different writing style while keeping exactly the same clinical findings and diagnoses.
Do not add, remove, or change any findings. Only change phrasing, word order, and sentence structure.

Original:
{text}

Rephrased version:"""

def rephrase_report(report_text: str, n_versions: int = 2) -> list[str]:
    versions = []
    for _ in range(n_versions):
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            messages=[{"role": "user", "content": REPHRASE_PROMPT.format(text=report_text)}]
        )
        versions.append(response.content[0].text.strip())
    return versions

# Load your training split
from src.dataset import load_splits
train_df, _, _ = load_splits("cxr_dataset.csv")

augmented_rows = []
for _, row in train_df.iterrows():
    rephrased = rephrase_report(row["report_text"], n_versions=2)
    for version in rephrased:
        new_row = row.copy()
        new_row["report_text"] = version
        new_row["report_id"] = row["report_id"] + f"_aug{len(augmented_rows)}"
        augmented_rows.append(new_row)

aug_df = pd.DataFrame(augmented_rows)
combined_df = pd.concat([train_df, aug_df], ignore_index=True)
combined_df.to_csv("cxr_dataset_augmented.csv", index=False)
print(f"Original: {len(train_df)} | Augmented: {len(combined_df)}")
```

**Cost estimate:** 2,600 reports × 2 rephrases × ~200 tokens/request × $5/1M tokens = ~$5 total.

**Quality check:** Before using the augmented data, sample 20 pairs and verify:
- The clinical meaning is preserved
- No findings were added or removed
- The text is grammatical

### Re-fine-tuning on augmented data

Two approaches:

**Option A: Train from scratch on combined data (recommended)**
```python
# In finetune.py, change:
CFG["csv_path"] = "cxr_dataset_augmented.csv"
```
Run the full training pipeline. The model never sees the original and augmented versions as "different" — they're just more data.

**Option B: Continue fine-tuning from a checkpoint**
If you already trained on original data and want to add augmented data:
```python
# Load your best checkpoint as the starting point
from peft import PeftModel
base = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", ...)
model = PeftModel.from_pretrained(base, "results/finetune/<run>/best_checkpoint")

# Then continue training with a lower learning rate
CFG["lr"] = 5e-5          # lower LR to avoid overwriting what was learned
CFG["epochs"] = 1         # only 1 more epoch — the model already knows radiology
CFG["csv_path"] = "cxr_dataset_augmented.csv"
```

Option B is useful when you want to add augmented data after already running a training run.

---

## Strategy 3: Synthetic image generation with RoentGen (high effort, highest reward)

RoentGen (Chambon et al., 2022) is a text-conditioned diffusion model specifically trained on chest X-rays. It can generate realistic CXR images from text descriptions.

**CLAUDE.md decision:** use RoentGen inference-only, don't retrain it.

### Workflow

```
1. Sample diverse CXR descriptions from your training reports
   (or write new ones: "bilateral pleural effusion", "right lower lobe pneumonia", etc.)

2. Generate synthetic CXR images using RoentGen
   text → [RoentGen diffusion] → synthetic image

3. Run a SOTA CXR model (e.g., CheXagent) to generate pseudo-labels
   synthetic image → [CheXagent] → pseudo-report

4. Quality filter: remove samples where pseudo-report has low confidence
   or contradicts the original text description

5. Add (synthetic_image, pseudo_report) pairs to training set
```

### Installing and using RoentGen

```python
# RoentGen is available on HuggingFace: StanfordAIMI/roentgen
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "StanfordAIMI/roentgen",
    torch_dtype=torch.float16
).to("cuda")

prompt = "Chest X-ray showing bilateral pleural effusion with blunting of costophrenic angles"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("synthetic_cxr.png")
```

### Generating pseudo-labels with CheXagent

```python
# CheXagent: StanfordAIMI/CheXagent
# It has a built-in report generation API — see its HuggingFace model card for exact usage
```

### When to use this strategy

This is worth the effort if:
- Your fine-tuned model struggles on rare findings (e.g., pneumothorax, cardiomegaly) that have few examples in NLM-CXR
- You have Colab Pro+ credits to burn (generation takes time)
- You want to demonstrate data augmentation in your portfolio write-up

Start with Strategy 1 + 2. Only add synthetic images if you have time and your model is still underfitting after 5+ epochs.

---

## Expected improvement from augmentation

| Training data | Expected ROUGE-L | Expected BERTScore F1 |
|---|---|---|
| Original ~2,600 reports | 0.20 – 0.28 | 0.84 – 0.87 |
| + Image augmentation | 0.22 – 0.30 | 0.85 – 0.88 |
| + Report rephrasing (~7,800) | 0.26 – 0.34 | 0.86 – 0.90 |
| + Synthetic images (~5,000 extra) | 0.28 – 0.36 | 0.87 – 0.91 |

Numbers are estimates based on similar work. The gains stack, but diminishing returns kick in quickly.

---

## Augmentation and data integrity rules

1. **Never augment the test set.** The test set must be 100% real, unmodified reports. Mixing augmented data into the test set invalidates your results.
2. **Keep the same split seed.** All augmented samples must come from the training split only.
3. **Track provenance.** Add an `is_augmented` column to your DataFrame so you can analyze whether the model performs differently on augmented vs original training samples.
4. **Run a sanity check before training.** Sample 5 augmented reports and confirm they still make clinical sense.
