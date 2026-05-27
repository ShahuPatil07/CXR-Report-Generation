"""
Week 2 — QLoRA fine-tuning of LLaVA-1.5-7B on NLM-CXR.

Run:
    python finetune.py

Checkpoints are written to results/finetune/<run_name>/
W&B is used for logging — set WANDB_PROJECT env var or edit CFG below.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.dataset import load_splits, CXRDataset
from src.metrics import evaluate_all, print_metrics

# ─── Config ─────────────────────────────────────────────────────────────────

CFG = {
    # Model
    "model_id":      "llava-hf/llava-1.5-7b-hf",

    # Data
    "csv_path":      str(ROOT / "cxr_dataset.csv"),
    "max_length":    512,

    # LoRA (rank 8 per CLAUDE.md — small dataset, conservative adapters)
    "lora_r":        8,
    "lora_alpha":    16,
    "lora_dropout":  0.05,
    "lora_targets":  ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training
    "epochs":             3,
    "per_device_batch":   2,
    "grad_accumulation":  8,    # effective batch = 2 * 8 = 16
    "lr":                 2e-4,
    "warmup_ratio":       0.03,
    "weight_decay":       0.01,
    "max_grad_norm":      1.0,
    "bf16":               True,

    # Eval & checkpointing
    "eval_every_steps":   200,
    "save_every_steps":   400,
    "n_val_samples":      100,   # sub-sample val for fast eval during training

    # W&B
    "wandb_project": os.environ.get("WANDB_PROJECT", "cxr-report-gen"),
    "run_name":      datetime.now().strftime("finetune_%Y%m%d_%H%M"),
}

OUTPUT_DIR = ROOT / "results" / "finetune" / CFG["run_name"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data ────────────────────────────────────────────────────────────────────

def build_loaders(processor):
    train_df, val_df, _ = load_splits(CFG["csv_path"])

    val_sample = val_df.sample(
        min(CFG["n_val_samples"], len(val_df)), random_state=42
    ).reset_index(drop=True)

    train_ds = CXRDataset(train_df, processor, mode="train",  max_length=CFG["max_length"])
    val_ds   = CXRDataset(val_sample, processor, mode="eval", max_length=CFG["max_length"])

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["per_device_batch"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, val_loader, val_sample


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(CFG["model_id"])
    model = LlavaForConditionalGeneration.from_pretrained(
        CFG["model_id"],
        quantization_config=bnb,
        device_map="auto",
    )

    # Prepare for k-bit training (adds gradient checkpointing, casts norms to fp32)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Freeze the ViT vision encoder — we only train the LLM adapters
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_targets"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, processor


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_validation(model, processor, val_loader, val_df, step):
    from PIL import Image as PILImage
    from src.dataset import INSTRUCTION, EVAL_TEMPLATE

    model.eval()
    hypotheses, references = [], []

    for batch_row in tqdm(val_df.itertuples(), total=len(val_df), desc="Val", leave=False):
        img = PILImage.open(batch_row.primary_image).convert("RGB")
        prompt = EVAL_TEMPLATE.format(instruction=INSTRUCTION)
        inputs = processor(
            text=prompt, images=img, return_tensors="pt"
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        gen = processor.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        hypotheses.append(gen)
        references.append(batch_row.report_text)

    metrics = evaluate_all(hypotheses, references, skip_bertscore=True)
    model.train()
    return metrics


# ─── Training loop ───────────────────────────────────────────────────────────

def train():
    import wandb

    wandb.init(project=CFG["wandb_project"], name=CFG["run_name"], config=CFG)

    model, processor = build_model()
    train_loader, val_loader, val_df = build_loaders(processor)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    total_steps = len(train_loader) * CFG["epochs"] // CFG["grad_accumulation"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=not CFG["bf16"])
    global_step = 0
    best_rougeL = 0.0

    for epoch in range(CFG["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16, enabled=CFG["bf16"]
            ):
                outputs = model(**batch)
                loss = outputs.loss / CFG["grad_accumulation"]

            if CFG["bf16"]:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            running_loss += loss.item() * CFG["grad_accumulation"]

            if (step + 1) % CFG["grad_accumulation"] == 0:
                if CFG["bf16"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CFG["max_grad_norm"]
                    )
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CFG["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / CFG["grad_accumulation"]
                running_loss = 0.0
                pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)
                wandb.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

                # Periodic validation
                if global_step % CFG["eval_every_steps"] == 0:
                    metrics = run_validation(model, processor, val_loader, val_df, global_step)
                    wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                    print(f"\n[Step {global_step}] Val metrics:")
                    print_metrics(metrics)

                    if metrics["rougeL"] > best_rougeL:
                        best_rougeL = metrics["rougeL"]
                        model.save_pretrained(OUTPUT_DIR / "best_checkpoint")
                        processor.save_pretrained(OUTPUT_DIR / "best_checkpoint")
                        print(f"  ✓ New best rougeL={best_rougeL:.4f} — checkpoint saved")

                # Periodic checkpoint
                if global_step % CFG["save_every_steps"] == 0:
                    ckpt_dir = OUTPUT_DIR / f"step_{global_step}"
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)

    # Final checkpoint
    model.save_pretrained(OUTPUT_DIR / "final_checkpoint")
    processor.save_pretrained(OUTPUT_DIR / "final_checkpoint")

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    print(f"\nTraining complete. Best val rougeL: {best_rougeL:.4f}")
    wandb.finish()


if __name__ == "__main__":
    train()
