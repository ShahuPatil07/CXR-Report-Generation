# `baseline.ipynb` — Zero-Shot Baseline Inference

## Purpose

Before fine-tuning, you need to know how bad the base model is. This gives you:
1. The "before" number in your results table
2. Proof that fine-tuning actually helped (and by how much)
3. A sanity check that your data pipeline and model loading work correctly

If the baseline is already surprisingly good, it means the base model has seen similar data and you have less to gain from fine-tuning.

---

## What each section does

### Section 2 — Load dataset

```python
train_df, val_df, test_df = load_splits(CSV_PATH, seed=SEED)
eval_df = test_df.sample(min(N_EVAL, len(test_df)), random_state=SEED)
```

You only evaluate on `N_EVAL=200` samples out of ~555 test reports. This is intentional — running inference on 555 samples would take ~45 minutes on an L4, and 200 is statistically sufficient to estimate mean metrics. If you want more robust numbers for the final write-up, set `N_EVAL=555` and run overnight.

### Section 3 — Load model (4-bit)

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=bnb_config)
```

The model is loaded in 4-bit even for inference. This is not strictly necessary for inference (you could load in fp16 if you have 16GB+ VRAM free), but it makes the notebook runnable on smaller GPUs. The quantization has minimal effect on output quality.

### Section 4 — Inference loop

```python
gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
generated = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
```

The key line: `output_ids` contains the full sequence (prompt + generation). You strip the prompt tokens by slicing from `inputs["input_ids"].shape[1]` onwards. If you forget this and decode the full sequence, you'll get "USER: Generate a radiology report... ASSISTANT: FINDINGS:..." in every output.

`do_sample=False` means greedy decoding — always pick the highest-probability token. This gives deterministic, reproducible output. For final evaluation, greedy is standard. For data augmentation (Week 3 DPO pairs), you'd want `do_sample=True` to get diverse outputs.

### Section 5 — Inspect examples

Look at these outputs carefully. Ask yourself:
- Is the model generating anything medically coherent?
- Is it hallucinating findings not visible in the image?
- Does it follow the FINDINGS / IMPRESSION structure?
- Is it repeating itself (a common failure mode)?

A typical zero-shot failure on CXR looks like:
```
Generated: "The image shows a chest X-ray of a patient. The lungs appear normal.
There are no obvious abnormalities. The heart size appears normal."
```
This sounds fine but is vague, non-specific, and would score near zero on RadGraph F1 because it has no specific anatomical findings.

### Section 6 — Save results

Results go to `results/baseline/`:
- `metrics.json` — all metric values + model name
- `predictions.csv` — per-sample: report_id, reference, generated

Keep these files. They are your "before" snapshot. After fine-tuning (Week 2) and DPO (Week 3), you will compare against these exact numbers.

---

## Running on Google Colab

1. Push your project to GitHub (or upload as a zip)
2. Clone/unzip on Colab
3. Mount Google Drive if you want to persist results:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run the pip install cell (uncomment it)
5. Set the runtime to GPU: Runtime → Change runtime type → A100 or L4

The model download (~14 GB for LLaVA-1.5-7B) happens automatically on first run. It will be cached in `~/.cache/huggingface/hub/`. On Colab, this cache is lost when the session ends — so if you restart, it re-downloads. To avoid this, copy the model to Drive after first download:
```python
!cp -r ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf /content/drive/MyDrive/
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `OutOfMemoryError: CUDA out of memory` | VRAM too small | Set `N_EVAL=50` to test, use A100 for full run |
| `OSError: llava-hf/llava-1.5-7b-hf not found` | No internet or wrong model ID | Check HuggingFace model page, ensure Colab has internet |
| `ModuleNotFoundError: bitsandbytes` | Not installed | Uncomment and run the pip install cell |
| Image file not found | Paths in CSV are absolute paths from your machine | Re-run `cxr_dataset.ipynb` on Colab to regenerate paths |

The last one is the most common issue when moving from local machine to Colab. The `primary_image` paths in the CSV point to your Windows filesystem. On Colab, re-extract the archives and re-run the data notebook to regenerate the CSV with Colab-relative paths.
