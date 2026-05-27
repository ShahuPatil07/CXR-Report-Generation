# `finetune.py` — QLoRA Fine-Tuning

## Architecture overview

```
[CXR image] → CLIP ViT-L → FROZEN (no gradients)
                              ↓
                    Projection MLP → trainable (not LoRA, just regular weights)
                              ↓
                    Vicuna-7B LLM → 4-bit quantized + LoRA adapters added
                              ↓
                    [generated report tokens]
```

Only ~4.2M parameters (0.06% of the model) are actually updated during training. Everything else is frozen.

---

## The `CFG` dictionary — what to tune and what not to touch

```python
CFG = {
    "model_id":         "llava-hf/llava-1.5-7b-hf",
    "lora_r":           8,
    "lora_alpha":       16,
    "lora_dropout":     0.05,
    "lora_targets":     ["q_proj", "k_proj", "v_proj", "o_proj"],
    "epochs":           3,
    "per_device_batch": 2,
    "grad_accumulation":8,
    "lr":               2e-4,
    "warmup_ratio":     0.03,
    ...
}
```

### Safe to tune
- `lr` — if training loss doesn't decrease, try `1e-4`. If it oscillates wildly, try `5e-5`.
- `epochs` — 3 is a good starting point. Watch the val ROUGE-L curve: stop when it plateaus.
- `lora_r` — raise to `16` if you add augmented data and the model underfits. Don't raise beyond 32 for this dataset size.
- `eval_every_steps` — reduce to `100` if you want more frequent checkpointing.
- `n_val_samples` — raise to `200` for more reliable validation estimates.

### Don't change unless you know why
- `lora_alpha` — keep it at `2 * lora_r`. This is a well-established convention.
- `warmup_ratio` — `0.03` is standard. Changing this rarely matters.
- `max_grad_norm` — `1.0` is the default everywhere. Leave it.
- `seed` — fix to 42 so train/val/test splits stay the same across runs.

---

## `build_model()` — step by step

### Step 1: 4-bit quantization

```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

The Vicuna-7B backbone is loaded with weights stored in 4-bit NormalFloat format. During the forward pass, each weight is dequantized to bf16 on-the-fly for computation, then discarded. This means:
- Storage: ~3.5 GB
- Computation: ~7 GB peak (due to activation tensors)
- Total VRAM needed for training: ~14–16 GB (with activations + gradients)

### Step 2: prepare for k-bit training

```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```

Three things happen internally:
1. LayerNorm weights are upcast to fp32. They must stay in fp32 — quantized LayerNorms cause NaN losses.
2. Gradient checkpointing is enabled. Instead of storing all intermediate activations in memory, PyTorch recomputes them during the backward pass. This trades ~30% speed for ~50% memory reduction on activations.
3. The input embeddings are set to require gradients (needed for LoRA to work through the embedding layer).

### Step 3: freeze the ViT

```python
for param in model.vision_tower.parameters():
    param.requires_grad = False
```

The CLIP ViT was trained on 400M image-text pairs. With only 3,700 training samples, updating it would cause catastrophic forgetting and likely hurt performance. Freezing it means the visual features stay general and stable — the LoRA adapters in the LLM learn to interpret them for the CXR domain.

### Step 4: apply LoRA

```python
lora_cfg = LoraConfig(
    r=8,                    # rank of the adapter matrices
    lora_alpha=16,          # scaling factor = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)
```

For each targeted weight matrix `W`, PEFT adds:
```
new_W = W + (B @ A) * (alpha / r)
```
Where `A` is initialized randomly and `B` is initialized to zero. This means at step 0, the LoRA adapter contributes nothing and the model outputs exactly what the pre-trained model would output. Training gradually nudges `A` and `B` to adapt the model to radiology language.

After `get_peft_model`, call `model.print_trainable_parameters()`. You'll see:
```
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

---

## `build_loaders()` — data pipeline

```python
train_ds = CXRDataset(train_df, processor, mode="train", max_length=512)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
```

`pin_memory=True` pins the CPU tensors to page-locked memory, which speeds up the CPU→GPU transfer. This matters when your bottleneck is data loading. `num_workers=2` means 2 parallel processes preload batches while the GPU computes. On Colab, don't set this above 4.

The val loader uses `batch_size=1` because validation runs `model.generate()` which doesn't support batching easily (outputs have variable length).

---

## `train()` — the training loop in detail

### Gradient accumulation

```python
loss = outputs.loss / CFG["grad_accumulation"]    # divide by 8
loss.backward()                                    # accumulate gradients

if (step + 1) % CFG["grad_accumulation"] == 0:    # every 8 steps
    optimizer.step()
    optimizer.zero_grad()
```

You can't fit batch_size=16 in memory. Instead, run 8 steps with batch_size=2, accumulating gradients without zeroing them. The gradient after 8 steps is identical to what you'd get from a single batch of 16 (because gradient is a sum, and summation is commutative). The division by `grad_accumulation` before `.backward()` normalizes the loss so the gradient magnitude is equivalent.

### bf16 and the GradScaler

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=CFG["bf16"]):
    outputs = model(**batch)
```

bfloat16 has the same dynamic range as fp32 (8-bit exponent) but less precision (7 mantissa bits vs 23). This means bf16 doesn't lose gradient signal due to underflow — so you don't need a `GradScaler`. The `GradScaler` code path is there only if you set `bf16=False` and use fp16 instead (on older GPUs that don't support bf16, like Tesla T4).

If you're on an A100 or L4, `bf16=True` is always the right choice. If you're on a T4, set `bf16=False`.

### Cosine LR schedule with warmup

The learning rate follows this curve over training:

```
LR
|       ___
|      /   \
|     /     \
|    /       \___________
|___/
    ^warmup  ^cosine decay
    3% steps   remaining
```

Warmup prevents the random LoRA adapters from making a destructively large update at step 0. The cosine tail gives the model time to converge without oscillating.

### Checkpoint selection

```python
if metrics["rougeL"] > best_rougeL:
    model.save_pretrained(OUTPUT_DIR / "best_checkpoint")
```

The best checkpoint is saved whenever val ROUGE-L improves. This is important: the final epoch's model is not necessarily the best — models often start to overfit on small datasets around epoch 2–3. Always use `best_checkpoint` for inference and for starting DPO (Week 3), not `final_checkpoint`.

---

## Loading a checkpoint for inference

After training, the checkpoint is a PEFT adapter (not a full model). Load it like this:

```python
from peft import PeftModel
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

base = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "results/finetune/<run>/best_checkpoint")
processor = AutoProcessor.from_pretrained("results/finetune/<run>/best_checkpoint")
```

The checkpoint directory contains only the LoRA adapter weights (~33 MB), not the full model. The base model is loaded fresh and the adapters are applied on top.

To merge the adapters permanently into the base weights (for deployment):
```python
model = model.merge_and_unload()
model.save_pretrained("results/merged_model")
```

---

## W&B monitoring — what to watch

| Metric | Healthy sign | Warning sign |
|---|---|---|
| `train/loss` | Steadily decreases from ~2.5 to ~1.0 | Flat or oscillating after warmup |
| `train/lr` | Follows cosine curve | Unexpected spikes |
| `val/rougeL` | Increases then plateaus | Decreasing (overfitting) |
| `val/rouge1` | Higher than rougeL | Inverted (means something is wrong with metrics) |

If training loss doesn't move at all after warmup, the most likely causes are:
1. LR too low (try 5e-4)
2. Labels all set to -100 — the `_find_subseq` call failed and the entire sequence is masked
3. Gradient not flowing (check `model.print_trainable_parameters()` output)
