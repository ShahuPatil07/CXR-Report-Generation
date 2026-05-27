# `src/dataset.py` — Data Loading and PyTorch Dataset

## What this file does

Three jobs:
1. Load and clean `cxr_dataset.csv` into a usable DataFrame
2. Split into train / val / test without leaking patient data
3. Provide a PyTorch `Dataset` that returns the right tensors for training vs inference

---

## `load_csv(csv_path)`

The CSV was saved by `cxr_dataset.ipynb` with list-valued columns (image paths, MeSH tags) stored as pipe-separated strings. `load_csv` reverses that:

```python
df["image_paths"] = df["image_paths"].apply(_parse_list_col)
# "path1|path2" → ["path1", "path2"]
```

It also:
- Builds `report_text` by combining the `findings` and `impression` columns into one structured string:
  ```
  FINDINGS: The lungs are clear bilaterally.
  IMPRESSION: No acute cardiopulmonary disease.
  ```
- Drops rows with no images or empty reports (they exist — some XML files had missing sections)
- Adds `primary_image` = the first image path per report. This is the PA (frontal) view, which is the standard view for report generation. Lateral views are available in `image_paths[1]` if you want to extend to multi-image input later.

---

## `load_splits(csv_path, val_frac=0.15, test_frac=0.15, seed=42)`

Returns `(train_df, val_df, test_df)`.

**Important: the split is done at the report level, not the image level.** Each NLM-CXR case can have a PA and a Lateral view. If you split by image, the same patient's PA view could end up in train and their Lateral view in test — this is data leakage. Splitting by report means both views of a patient always go to the same split.

Default proportions on ~3,700 usable reports:
- Train: ~2,600 reports
- Val: ~555 reports
- Test: ~555 reports

The `seed=42` makes splits reproducible. Never change this after your first training run, or your test set becomes invalid.

---

## `CXRDataset`

### Constructor

```python
CXRDataset(df, processor, mode="train", max_length=512)
```

- `processor` — the HuggingFace processor for your model (handles both image preprocessing and tokenization)
- `mode` — `"train"` or `"eval"`. The difference matters a lot (see below).
- `max_length=512` — sequences longer than this are truncated. 512 covers ~95% of reports in this dataset. If you see warnings about truncation during training, raise this to 768 at the cost of more memory.

### `__getitem__` in train mode

Returns a dict with keys: `input_ids`, `attention_mask`, `pixel_values`, `labels`.

The text fed to the processor looks like:

```
USER: <image>
Generate a radiology report for this chest X-ray. ASSISTANT: FINDINGS: ... IMPRESSION: ...\n</s>
```

The `</s>` at the end is the EOS token — the model needs to learn when to stop generating.

### `__getitem__` in eval mode

The text stops at `ASSISTANT:` (no report). The model generates everything after that during `.generate()`.

```
USER: <image>
Generate a radiology report for this chest X-ray. ASSISTANT:
```

### Label masking — the most important concept

The `labels` tensor controls where the loss is computed. In PyTorch, any position with label `-100` is skipped by `CrossEntropyLoss`.

```python
labels = inputs["input_ids"].clone()
boundary = _find_subseq(ids, self._assistant_ids)
labels[: boundary + len(self._assistant_ids)] = -100   # mask the instruction
labels[labels == pad_id] = -100                        # mask padding
```

Visually:

```
Token: USER  :  <img>  Generate  ...  ASSISTANT  :  FINDINGS  :  lungs  ...  </s>  [PAD] [PAD]
Label: -100  -100 -100    -100   -100    -100    -100   8621    25   4913  ...   2    -100   -100
```

**Why this matters:** the model is a next-token predictor. If you didn't mask the instruction, it would spend most of its gradient budget learning to predict "Generate a radiology report" — which is always identical. By masking it, 100% of the learning signal comes from the actual report content.

The `_find_subseq` helper searches for the token IDs of `"ASSISTANT:"` in the full sequence. It's a linear scan, which is fine for sequences of 512 tokens.

### Extending this for multi-image input

If you want to train with both PA and Lateral views, you would:
1. Pass a list of images to `processor(images=[img_pa, img_lat], ...)` — LLaVA supports this
2. Put two `<image>` tokens in your prompt: `"USER: <image><image>\n{instruction} ASSISTANT: ..."`
3. Everything else stays the same

This is a natural Week 4 extension for the ablation study.

---

## Constants you may want to change

| Constant | Location | Default | When to change |
|---|---|---|---|
| `INSTRUCTION` | line 13 | `"Generate a radiology report..."` | If you want to experiment with different prompts |
| `max_length` | `__init__` arg | `512` | Raise to 768 if truncation warnings appear |
| `val_frac`, `test_frac` | `load_splits` args | `0.15` | Don't change after first run |
| `seed` | `load_splits` arg | `42` | Don't change after first run |
